from collections import defaultdict
import os

# Initialize a defaultdict with lists as the default value type
image_data = defaultdict(list)

# Define the path to your text file
text_file_path = "Sim 2.23 - part2.txt"

# Check if the file exists
if os.path.exists(text_file_path):
    # Open and read the text file
    with open(text_file_path, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Split the line by comma
            image_name, coords = line.split(',', 1)
            
            # Strip any leading or trailing whitespaces from the image name and coordinates
            image_name = image_name.strip()
            coords = coords.strip()
            
            # Append the coordinates to the list associated with the image name
            image_data[image_name].append(coords)
else:
    print(f"File {text_file_path} not found.")

# Save the organized data into separate text files
output_folder = "Sim 1.23a - part 2 - copy/Labels_txt"

# Make sure output folder exists, create it if it doesn't
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for image_name, coords_list in image_data.items():
    output_file_path = os.path.join(output_folder, f"{image_name.replace('.png', '')}.txt")
    
    with open(output_file_path, 'w') as file:
        for coords in coords_list:
            file.write(f"{coords}\n")
