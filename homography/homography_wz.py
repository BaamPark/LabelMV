# This script use homography to associate bounding boxes across views
# These locations are read from image with index 52.
# Location #1: The foot of the metal table that is closest to the bed foot.
# Location #2: The wheel of the white table near the door and beside the wall. Closer to providers. Farther from the black line on the ground.
# Location #3: Tiny table with a black panel behind Aaron. The foot closer the the door. 
# Location #4: Connet Dynan's left foot and location #2. It cross with the brown circle on the ground.
from logger_config import logger
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

class HomographyAssociator:
    def __init__(self, src_points, dst_points_1, dst_points_2):
        """
        Initialize with reference points for homography calculation for two mappings.
        
        :param src_points: Points from the source image for homography calculation.
        :param dst_points_1: Points from the first destination image for homography calculation.
        :param dst_points_2: Points from the second destination image for homography calculation.
        """
        self.src_points = np.array(src_points, dtype='float32')
        self.dst_points_1 = np.array(dst_points_1, dtype='float32')
        self.dst_points_2 = np.array(dst_points_2, dtype='float32')
        self.homography_matrix_1 = self.calculate_homography(self.src_points, self.dst_points_1)
        self.homography_matrix_2 = self.calculate_homography(self.src_points, self.dst_points_2)

    def calculate_homography(self, src_pts, dst_pts):
        """Calculate homography matrix from src_points to dst_points."""
        H, _ = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
        return H

    def compute_cost_matrix(self, src_points, dst_points):
        """Compute the cost matrix for the distances between transformed source points and destination points."""
        cost_matrix = []
        for src_point in src_points:
            distances = [distance.euclidean(src_point, dst_point) for dst_point in dst_points]
            cost_matrix.append(distances)
        return np.array(cost_matrix)
    
    def get_key_point(self, bbox, mode='bottom'):
        """ Get a point for homography transformation """
        xmin, ymin, xmax, ymax = bbox
        if mode == 'bottom':
            key_point = ((xmin + xmax) // 2, ymax)
        elif mode == 'centroid':
            key_point = ((xmin + xmax) // 2, (ymin + ymax) // 2)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return key_point
    
    def apply_homography(self, point, homography_matrix):
        """Apply homography to transform a point."""
        src_point = np.array([[point[0], point[1], 1]]).T
        print(homography_matrix)
        print(src_point)
        dst_point = np.dot(homography_matrix, src_point)
        print(dst_point)
        dst_point /= dst_point[2]  # Normalize by the third coordinate
        print(dst_point)
        return int(dst_point[0]), int(dst_point[1])

    def associate_bboxes_direct(self, src_bboxes, dst_bboxes, homography_matrix):
        """Associate bboxes between the src and dst using the Hungarian algorithm to minimize total distance."""
    
        # Prepare the cost matrix (distances between each pair of source and destination points)
        cost_matrix = []
        transformed_points = []
        
        key_point_mode = 'bottom'
        # key_point_mode = 'centroid'

        for src_bbox in src_bboxes:
            # Transform the bottom-center point of the source bounding box
            # bottom_center_src = self.get_key_point(src_bbox, mode='bottom')
            # Transform the centroid point of the source bounding box # Not much difference.
            bottom_center_src = self.get_key_point(src_bbox, mode=key_point_mode)

            transformed_point = self.apply_homography(bottom_center_src, homography_matrix)
            transformed_points.append(transformed_point)
            
            # Calculate distances to all destination bounding boxes
            distances = []
            for dst_bbox in dst_bboxes:
                bottom_center_dst = self.get_key_point(dst_bbox)
                dist = distance.euclidean(transformed_point, bottom_center_dst)
                distances.append(dist)
            cost_matrix.append(distances)

        # Use the Hungarian algorithm to minimize the total distance and find the optimal assignment
        # In short, Hungarian algorithm finds the optimal assignment that minimizes the total cost
        # The matrix value represents the cost of assigning the source point (rows) to the destination point (cols)
        # Here the cost is the distance.
        cost_matrix = np.array(cost_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Create the associations based on the optimal assignment
        associations = []
        for i, j in zip(row_ind, col_ind):
            src_bbox = src_bboxes[i]
            dst_bbox = dst_bboxes[j]
            transformed_point = transformed_points[i]
            associations.append((src_bbox, dst_bbox, self.get_key_point(src_bbox, key_point_mode), transformed_point))

        return associations

    def associate_bboxes_greedy(self, src_bboxes, dst_bboxes, homography_matrix, view, threshold=1e-2):
        """Associate bboxes between the src and dst using an iterative exclusion method and Hungarian algorithm."""
        print('=> Greedy Association')

        # Transform source points and calculate initial cost matrix
        transformed_points = [self.apply_homography(self.get_key_point(src_bbox), homography_matrix) for src_bbox in src_bboxes]
        dst_points = [self.get_key_point(dst_bbox) for dst_bbox in dst_bboxes]
        org_src, org_dst = len(src_bboxes), len(dst_bboxes)

        iterations = 0
        while True:
            # print(f'** ITERATION {iterations} ** Remaining Bboxes: src=[{len(src_bboxes)}/{org_src}] dst=[{len(dst_bboxes)}/{org_dst}]')

            # Step 1: Run Hungarian algorithm for initial associations and remove unmatched bboxes
            cost_matrix = self.compute_cost_matrix(transformed_points, dst_points)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Filter out unmatched bounding boxes
            src_bboxes = [src_bboxes[i] for i in row_ind]
            dst_bboxes = [dst_bboxes[j] for j in col_ind]
            transformed_points = [transformed_points[i] for i in row_ind]
            dst_points = [dst_points[j] for j in col_ind]
            
            # print(f'** ITERATION {iterations} ** Remaining Bboxes: src=[{len(src_bboxes)}/{org_src}] dst=[{len(dst_bboxes)}/{org_dst}]')
            # Redo it again for the updated mapping index.
            cost_matrix = self.compute_cost_matrix(transformed_points, dst_points)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Compute the initial cost of the full set of associations
            initial_cost = cost_matrix[row_ind, col_ind].sum() / len(row_ind)
            original_pairs = set(zip(row_ind, col_ind))
            
            # Stop condition: if there is only one bounding box left in either src or dst, we stop
            if len(src_bboxes) <= 1 or len(dst_bboxes) <= 1:
                print("Stopping early to retain at least one bounding box in each set.")
                break

            # Step 2: Evaluate each bounding box for removal
            cost_differences = []
            
            # Check each source bounding box for potential removal
            for i in range(len(src_bboxes)):
                temp_transformed_points = np.delete(transformed_points, i, axis=0)
                temp_cost_matrix = self.compute_cost_matrix(temp_transformed_points, dst_points)
                temp_row_ind, temp_col_ind = linear_sum_assignment(temp_cost_matrix)

                # Adjust temp_row_ind to reflect the original indices
                temp_row_ind_temp = np.copy(temp_row_ind)
                for order in range(i, len(temp_row_ind)):
                    temp_row_ind_temp[order] += 1

                temp_pairs = set(zip(temp_row_ind_temp, temp_col_ind))

                if not temp_pairs.issubset(original_pairs):
                    # print(f'** ITERATION {iterations} ** Checking src bbox {i}')
                    # print(f'\t {original_pairs} | {row_ind} {col_ind}')
                    # print(f'\t {temp_pairs} | {temp_row_ind} {temp_col_ind}')
                    temp_cost = temp_cost_matrix[temp_row_ind, temp_col_ind].sum() / len(temp_row_ind)
                    cost_differences.append((initial_cost - temp_cost, 'src', i))

            # Check each destination bounding box for potential removal
            for j in range(len(dst_bboxes)):
                temp_dst_points = np.delete(dst_points, j, axis=0)
                temp_cost_matrix = self.compute_cost_matrix(transformed_points, temp_dst_points)
                temp_row_ind, temp_col_ind = linear_sum_assignment(temp_cost_matrix)

                # Adjust temp_col_ind to reflect the original indices
                temp_col_ind_temp = np.copy(temp_col_ind)
                for order in range(len(temp_col_ind_temp)):
                    if temp_col_ind_temp[order] >= j:
                        temp_col_ind_temp[order] += 1

                temp_pairs = set(zip(temp_row_ind, temp_col_ind_temp))

                # If associations remain unchanged, mark it as "good"
                if not temp_pairs.issubset(original_pairs):
                    # print(f'** ITERATION {iterations} ** Checking dst bbox {j}')
                    # print(f'\t {original_pairs}')
                    # print(f'\t {temp_pairs}')
                    temp_cost = temp_cost_matrix[temp_row_ind, temp_col_ind].sum() / len(temp_row_ind)
                    cost_differences.append((initial_cost - temp_cost, 'dst', j))
            
            # Stop condition: if no bounding boxes are identified for removal, we stop
            if not cost_differences:
                print("No bounding boxes to remove, stopping.")
                break

            # Remove the bounding box with the largest cost difference
            max_difference, bbox_type, index = max(cost_differences, key=lambda x: x[0])

            # Stop if the maximum difference is below the threshold
            if max_difference <= threshold:
                print("Max difference below threshold, stopping.")
                break
            
            # Exclude the identified bounding box
            if bbox_type == 'src':
                src_bboxes.pop(index)
                transformed_points.pop(index)
                # print(f'** ITERATION {iterations} ** Removing src bbox {index}')
            elif bbox_type == 'dst':
                dst_bboxes.pop(index)
                dst_points.pop(index)
                # print(f'** ITERATION {iterations} ** Removing dst bbox {index}')
            

            ######################################################################
            # Draw the current state of the bounding boxes for debugging
            # Final assignment after exclusions
            ######################################################################
            # cost_matrix = self.compute_cost_matrix(transformed_points, dst_points)
            # row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # # Create the associations based on the final assignment
            # asso = []
            # for i, j in zip(row_ind, col_ind):
            #     src_bbox = src_bboxes[i]
            #     dst_bbox = dst_bboxes[j]
            #     transformed_point = transformed_points[i]
            #     asso.append((src_bbox, dst_bbox, self.get_key_point(src_bbox), transformed_point))

            # src_img = cv2.imread('./data/Duration_Simulation_5_Scenarios/10.29_scenario_2_trauma_1/10.29_scenario_2_trauma_1_above_bed/Frames/10.29_scenario_2_trauma_1_above_bed_0.png')
            # if view == 1:
            #     dst_img_1 = cv2.imread('./data/Duration_Simulation_5_Scenarios/10.29_scenario_2_trauma_1/10.29_scenario_2_trauma_1_foot_of_bed/Frames/10.29_scenario_2_trauma_1_foot_of_bed_0.png')
            # else:
            #     dst_img_1 = cv2.imread('./data/Duration_Simulation_5_Scenarios/10.29_scenario_2_trauma_1/10.29_scenario_2_trauma_1_foot_of_bed#2/Frames/10.29_scenario_2_trauma_1_foot_of_bed#2_0.png')

            # # Draw source and destination
            # src_img_with_bboxes_1 = self.draw_bboxes_and_points(src_img.copy(), asso, draw_src=True)
            # dst_img_with_bboxes_1 = self.draw_bboxes_and_points(dst_img_1.copy(), asso, draw_src=False)

            # white_margin = 255 * np.ones((src_img_with_bboxes_1.shape[0], 10, 3), dtype=np.uint8)

            # row1 = np.hstack((src_img_with_bboxes_1, white_margin, dst_img_with_bboxes_1))

            # # Save the combined result image if needed
            # save_path = f"./pic/debugging/combined_view_0_view{view}_iter{iterations}.png"
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # cv2.imwrite(save_path, row1)
            ######################################################################
            ######################################################################
            iterations += 1

        # Final assignment after exclusions
        cost_matrix = self.compute_cost_matrix(transformed_points, dst_points)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create the associations based on the final assignment
        associations = []
        for i, j in zip(row_ind, col_ind):
            src_bbox = src_bboxes[i]
            dst_bbox = dst_bboxes[j]
            transformed_point = transformed_points[i]
            associations.append((src_bbox, dst_bbox, self.get_key_point(src_bbox), transformed_point))
        
        return associations
    
    def associate_bboxes_nearest_k(self, src_bboxes, dst_bboxes, homography_matrix, k=2):
        """Associate bounding boxes between the src and dst views using k-NN for outlier detection and the Hungarian algorithm for optimal assignment."""

        # Transform the source points and calculate distances
        transformed_points = []
        key_point_mode = 'bottom'  # Set to 'centroid' if centroid is preferred

        for src_bbox in src_bboxes:
            # Get key point from the source bounding box and transform it
            key_point_src = self.get_key_point(src_bbox, mode=key_point_mode)
            transformed_point = self.apply_homography(key_point_src, homography_matrix)
            transformed_points.append(transformed_point)

        # Calculate distances between each transformed point and all destination points
        dst_points = [self.get_key_point(dst_bbox, mode=key_point_mode) for dst_bbox in dst_bboxes]
        distance_matrix = self.compute_cost_matrix(transformed_points, dst_points)

        # Identify nearest neighbors
        nearest_neighbors_set = set()
        for distances in distance_matrix:
            # Find indices of the two nearest destination points for each transformed point
            nearest_indices = np.argsort(distances)[:k]
            # Add these points to the set of nearest neighbors
            for idx in nearest_indices:
                nearest_neighbors_set.add(idx)

        # Filter destination bounding boxes based on nearest neighbor indices
        filtered_dst_bboxes = [dst_bboxes[i] for i in nearest_neighbors_set]
        filtered_dst_points = [dst_points[i] for i in nearest_neighbors_set]

        # Recompute the distance matrix for the filtered points
        filtered_distance_matrix = self.compute_cost_matrix(transformed_points, filtered_dst_points)

        # Run the Hungarian algorithm for optimal assignment on the filtered set
        row_ind, col_ind = linear_sum_assignment(filtered_distance_matrix)

        # Create associations based on the optimal assignment
        associations = []
        for src_idx, dst_idx in zip(row_ind, col_ind):
            src_bbox = src_bboxes[src_idx]
            dst_bbox = filtered_dst_bboxes[dst_idx]
            transformed_point = transformed_points[src_idx]
            associations.append((src_bbox, dst_bbox, self.get_key_point(src_bbox, key_point_mode), transformed_point))

        return associations
    
    def associate_bboxes_nearest_k_ver2(self, src_bboxes, dst_bboxes, homography_matrix, k=2):
        logger.info("<==associate_bboxes_nearest_k_ver2 function is called==>")
        transformed_points = []
        key_point_mode = 'bottom'

        for src_bbox in src_bboxes:
            key_point_src = self.get_key_point(src_bbox, mode=key_point_mode)
            transformed_point = self.apply_homography(key_point_src, homography_matrix)
            transformed_points.append(transformed_point)

        dst_points = [self.get_key_point(dst_bbox, mode=key_point_mode) for dst_bbox in dst_bboxes]
        distance_matrix = self.compute_cost_matrix(transformed_points, dst_points)
        logger.info(f"dst_points before KNN: {dst_points}")

        nearest_neighbors_set = set()
        for distances in distance_matrix:
            nearest_indices = np.argsort(distances)[:k]
            for idx in nearest_indices:
                nearest_neighbors_set.add(idx)

        filtered_dst_points = [dst_points[i] for i in nearest_neighbors_set]
        nearest_neighbors_list = list(nearest_neighbors_set)
        logger.info(f"dst_points after KNN: {filtered_dst_points}")
        filtered_distance_matrix = self.compute_cost_matrix(transformed_points, filtered_dst_points)

        row_ind, col_ind = linear_sum_assignment(filtered_distance_matrix)

        assignments = [[row, nearest_neighbors_list[col]] for row, col in zip(row_ind, col_ind)]
        logger.info(f"assignments: {assignments}")
        return assignments
    

    def associate_bboxes_ver2(self, src_bboxes, dst_bboxes, homography_matrix, k=2):
        logger.info("<==associate_bboxes_nearest_k_ver2 function is called==>")
        transformed_points = []
        key_point_mode = 'bottom'

        for src_bbox in src_bboxes:
            key_point_src = self.get_key_point(src_bbox, mode=key_point_mode)
            transformed_point = self.apply_homography(key_point_src, homography_matrix)
            transformed_points.append(transformed_point)

        dst_points = [self.get_key_point(dst_bbox, mode=key_point_mode) for dst_bbox in dst_bboxes]
        distance_matrix = self.compute_cost_matrix(transformed_points, dst_points)
        logger.info(f"dst_points before KNN: {dst_points}")

        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        assignments = [[row, col] for row, col in zip(row_ind, col_ind)]
        logger.info(f"assignments: {assignments}")
        return assignments
    
    def associate_bboxes_OpenPose(self, src_bboxes, dst_bboxes, homography_matrix, threshold=1e-2):
        pass


    def draw_bboxes_and_points(self, image, associations, draw_src):
        """Draw bounding boxes and points on the image."""
        for i, (src_bbox, dst_bbox, src_point, dst_point) in enumerate(associations):
            if draw_src:
                # Draw source bounding box and point
                xmin, ymin, xmax, ymax = src_bbox
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)  # Thicker line for border
                # Draw source point in red
                cv2.circle(image, src_point, 10, (0, 0, 255), -1)  # Red point
                cv2.putText(image, f'Source View', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3) 
                cv2.putText(image, f"ID {i}", (xmin + 5, ymin + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)  # Bigger font size and thickness
            else:
                # Draw destination bounding box and transformed point
                xmin, ymin, xmax, ymax = dst_bbox
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 4)  # Thicker line for border
                # Draw transformed point in red
                cv2.circle(image, dst_point, 10, (0, 0, 255), -1)  # Red point
                cv2.putText(image, f'Destination View', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)  # White text
                cv2.putText(image, f"ID {i}", (xmin + 5, ymin + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)  # Bigger font size and thickness

        return image

    def process_association_above_to_foot1(self, src_bboxes, dst_bboxes):
        associations = self.associate_bboxes_ver2(src_bboxes, dst_bboxes, self.homography_matrix_1)
        # associations = self.associate_bboxes_nearest_k_ver2(src_bboxes, dst_bboxes, self.homography_matrix_1)
        return associations
    
    def process_associations(self, src_bboxes, dst_bboxes_1, dst_bboxes_2):
        # Choose which way to associate bboxes
        ##################################################################
        # Direct Association
        # associations_1 = self.associate_bboxes_direct(src_bboxes, dst_bboxes_1, self.homography_matrix_1)
        # associations_2 = self.associate_bboxes_direct(src_bboxes, dst_bboxes_2, self.homography_matrix_2)
        # Greedy Association
        # associations_1 = self.associate_bboxes_greedy(src_bboxes, dst_bboxes_1, self.homography_matrix_1, view=1)
        # associations_2 = self.associate_bboxes_greedy(src_bboxes, dst_bboxes_2, self.homography_matrix_2, view=2)
        # KNN Association
        associations_1 = self.associate_bboxes_nearest_k(src_bboxes, dst_bboxes_1, self.homography_matrix_1)
        associations_2 = self.associate_bboxes_nearest_k(src_bboxes, dst_bboxes_2, self.homography_matrix_2)

        return associations_1, associations_2
    
    def show_results(self, images_path, video_name, img_id, associations_1, associations_2, show=False):
        # Load images
        src_img = cv2.imread(images_path[0])
        dst_img_1 = cv2.imread(images_path[1])
        dst_img_2 = cv2.imread(images_path[2])

        # Draw source and destination
        src_img_with_bboxes_1 = self.draw_bboxes_and_points(src_img.copy(), associations_1, draw_src=True)
        dst_img_with_bboxes_1 = self.draw_bboxes_and_points(dst_img_1.copy(), associations_1, draw_src=False)
        src_img_with_bboxes_2 = self.draw_bboxes_and_points(src_img.copy(), associations_2, draw_src=True)
        dst_img_with_bboxes_2 = self.draw_bboxes_and_points(dst_img_2.copy(), associations_2, draw_src=False)

        # Create a white margin (10 pixels wide)
        margin_width = 10
        white_margin = 255 * np.ones((src_img_with_bboxes_1.shape[0], margin_width, 3), dtype=np.uint8)

        row1 = np.hstack((src_img_with_bboxes_1, white_margin, dst_img_with_bboxes_1))
        row2 = np.hstack((src_img_with_bboxes_2, white_margin, dst_img_with_bboxes_2))

        # Add white margin between rows
        row_margin = 255 * np.ones((margin_width, row1.shape[1], 3), dtype=np.uint8)

        # Combine rows into a grid (3 images with margins)
        combined_img = np.vstack((row1, row_margin, row2))

        # Save the combined result image if needed
        save_path = f"./pic/{video_name+'_homography_KNN'}/combined_view_{img_id}.png"
        # save_path = f"./pic/{video_name+'_homography_greedy'}/combined_view_{img_id}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, combined_img)

        if show:
            cv2.imshow("Source and Destination Views", combined_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# Helper function to parse bounding boxes
def parse_bboxes_from_xml(xml_file):
    """Parse bounding boxes from XML file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bboxes.append((xmin, ymin, xmax, ymax))  # Entire bounding box coordinates
    return bboxes

def get_homography_associator_object():
    points_src = [(1273, 173), (273, 148), (1309, 515), (435, 358)]  # view[0]
    points_dst_1 = [(768, 521), (1526, 566), (793, 396), (1315, 454)]  # view[1]
    points_dst_2 = [(1068, 334), (476, 1068), (1327, 412), (844, 1019)]  # view[2]
    associator = HomographyAssociator(points_src, points_dst_1, points_dst_2)
    return associator


# Example usage
if __name__ == "__main__":
    img_id = [i for i in range(100)]
    # img_id = [0]
    
    # Video One
    data_name = 'R2PPE'
    video_name = 'bed_sim_3'
    view_name = [video_name, video_name+'_foot_1', video_name+'_foot_2']

    # Video Two
    # data_name = 'Duration_Simulation_5_Scenarios'
    # video_name = '10.29_scenario_2_trauma_1'
    # view_name = [video_name+'_above_bed', video_name+'_foot_of_bed', video_name+'_foot_of_bed#2']
    
    points_src = [(1273, 173), (273, 148), (1309, 515), (435, 358)]  # view[0]
    points_dst_1 = [(768, 521), (1526, 566), (793, 396), (1315, 454)]  # view[1]
    points_dst_2 = [(1068, 334), (476, 1068), (1327, 412), (844, 1019)]  # view[2]

    # Create the associator for all three views
    associator = HomographyAssociator(points_src, points_dst_1, points_dst_2)

    for i in img_id:
        img_path = [f'./data/{data_name}/{video_name}/{view}/Frames/{view}_{i}.png' for view in view_name]
        lbl_path = [f'./data/{data_name}/{video_name}/{view}/Labels/{view}_{i}.xml' for view in view_name]

        print(f'=> Working on {img_path[0]}')
        # Parse bounding boxes from XML files
        src_bboxes = parse_bboxes_from_xml(lbl_path[0])
        dst_bboxes_1 = parse_bboxes_from_xml(lbl_path[1])
        dst_bboxes_2 = parse_bboxes_from_xml(lbl_path[2])
        # Process associations for both mappings
        associations_1, associations_2 = associator.process_associations(src_bboxes, dst_bboxes_1, dst_bboxes_2)
        
        break

        # Show results
        associator.show_results(img_path, video_name, i, associations_1, associations_2, show=False)
    


##################################################
'''
Initially, I am using the middle point of the bottom of the bounding box 
for both homography transformation in src view and association in the dst view.
=> ./pic/bed_sim_3_homography
=> ./pic/bed_sim_3_homography_KNN
=> ./pic/10.29_scenario_2_trauma_1_homography
=> ./pic/10.29_scenario_2_trauma_1_homography_2
I tried using the centroid of the bbox in the src view and also the middle bottom point in the dst view.
This is because the overhead view likely to have the foot point at the centroid of the bbox. 
=> ./pic/10.29_scenario_2_trauma_1_homography_greedy
Tried using greedy algorithm. Results seem to be not good.
=> ./pic/10.29_scenario_2_trauma_1_homography_KNN
For any src point, check the nearest 2 points in the dst view. Remove the dst point that is not in the nearest 2 points of any src point.
'''