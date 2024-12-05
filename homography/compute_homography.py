import cv2
import numpy as np
import pickle

def main():
    compute_homography()
    mark_and_map("homography/view1.jpg", "homography/view2.jpg")

# Points from image1 and image2
def compute_homography():
    points_image1 = np.array([
        [1192, 328],
        [363, 671],
        [218, 277],
        [1404, 935]
    ], dtype=np.float32)

    points_image2 = np.array([
        [827, 449],
        [1307, 341],
        [1492, 465],
        [689, 293]
    ], dtype=np.float32)

    # Compute the homography matrix
    homography_matrix, status = cv2.findHomography(points_image1, points_image2, method=cv2.RANSAC)

    # Compute the inverse homography matrix
    inverse_homography_matrix = np.linalg.inv(homography_matrix)

    # Print the results
    print("Homography Matrix:")
    print(homography_matrix)
    with open("homography/homography_matrix.pkl", "wb") as f:
        pickle.dump(homography_matrix, f)

    print("\nInverse Homography Matrix:")
    print(inverse_homography_matrix)
    with open("homography/inverse_homography_matrix.pkl", "wb") as f:
        pickle.dump(inverse_homography_matrix, f)

def mark_and_map(img1_path, img2_path):
    image1 = cv2.imread(img1_path)  # Replace with your image path
    image2 = cv2.imread(img2_path)  # Replace with your image path

    # Load homography matrices
    pickle_file_path = 'homography/homography_matrices.pkl'  # Replace with your pickle file path
    with open(pickle_file_path, 'rb') as file:
        homography_matrix = pickle.load(file)  # Assuming the homography matrix is directly stored

    # Callback function for mouse click
    def mark_and_map(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Mark dot on image1
            marked_image1 = image1.copy()
            cv2.circle(marked_image1, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Image 1', marked_image1)

            # Map the point using the homography matrix
            point = np.array([[x, y, 1]], dtype=np.float32).T
            mapped_point = np.dot(homography_matrix, point)
            mapped_point /= mapped_point[2]  # Normalize to homogeneous coordinates
            mapped_x, mapped_y = int(mapped_point[0]), int(mapped_point[1])

            # Mark mapped point on image2
            marked_image2 = image2.copy()
            cv2.circle(marked_image2, (mapped_x, mapped_y), 5, (0, 255, 0), -1)
            cv2.imshow('Image 2', marked_image2)

    # Display image1 and set mouse callback
    cv2.imshow('Image 1', image1)
    cv2.setMouseCallback('Image 1', mark_and_map)

    # Keep the windows open until user closes them
    cv2.imshow('Image 2', image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()