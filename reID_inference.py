import numpy as np
import pickle

def get_center_bottom(ltwh: list, homography: np.array, apply_homography=False):
    left, top, width, height = ltwh
    center_x = left + width / 2
    bottom_y = top + height
    target_point = [center_x, bottom_y]

    if apply_homography:
        target_point_homogeneous = np.array([*target_point, 1])
        mapped_target_point = np.dot(homography, target_point_homogeneous)
        mapped_target_point /= mapped_target_point[2]  # Normalize to make it homogeneous
        return mapped_target_point[:2]
    
    return np.array(target_point)


def compute_homography_distance(bbox_list_from_view1: list, bbox_list_from_view2: list):
    if len(bbox_list_from_view1) != len(bbox_list_from_view2):
        raise ValueError("bbox_list_from_view1 and bbox_list_from_view2 should have the same length.")
    
    homography = pickle.load(open('homography\homography_matrix.pkl', 'rb'))
    new_bbox_list_from_view1 = [
        get_center_bottom(bbox_from_view1, homography, apply_homography=True)
        for bbox_from_view1 in bbox_list_from_view1
    ]
    new_bbox_list_from_view2 = [
        get_center_bottom(bbox_from_view2, homography, apply_homography=False)
        for bbox_from_view2 in bbox_list_from_view2
    ]

    distance_matrix = np.zeros((len(new_bbox_list_from_view1), len(new_bbox_list_from_view2)))
    for i, point1 in enumerate(new_bbox_list_from_view1):
        for j, point2 in enumerate(new_bbox_list_from_view2):
            distance_matrix[i, j] = np.linalg.norm(point1 - point2)

    return distance_matrix


if __name__ == "__main__":
    # Mock bounding box lists for testing
    bbox_list_from_view1 = [
        [1192, 328, 7, 7],  # [left, top, bottom, right]
        [363, 671, 8, 4],
        [218, 277, 6, 1],
        [1404, 935, 6, 4]
    ]
    bbox_list_from_view2 = [
        [827, 449, 2, 0],
        [1307, 341, 2, 4],
        [1492, 465, 1, 1],
        [689, 293, 2, 0]
    ]

    distance_matrix = compute_homography_distance(bbox_list_from_view1, bbox_list_from_view2)
    print("Distance Matrix:")
    print(distance_matrix)