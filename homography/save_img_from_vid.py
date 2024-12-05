import cv2

# Step 1: Open the video
video_path = "samples_vid\VP-view2.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# Step 2: Read the first frame
ret, frame = cap.read()

if ret:
    # Save the first frame
    output_image_path = "homography/view2.jpg"  # Save path for the first frame
    cv2.imwrite(output_image_path, frame)
    print(f"First frame saved as {output_image_path}")
else:
    print("Error: Cannot read the first frame.")

# Release the video capture object
cap.release()
