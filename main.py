import cv2
import numpy as np
import os
from glob import glob

# Set the input and output folders
input_folder = '/storage/emulated/0/Download/'
output_folder = '/storage/emulated/0/OUTPUT/'
watermark_template_path = '/storage/emulated/0/Download/watermark_template.png'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get all image files from the input folder with specified extensions
image_extensions = ['*.jpeg', '*.jpg', '*.png']
image_files = []
for ext in image_extensions:
    image_files.extend(glob(os.path.join(input_folder, ext)))

# Load the watermark template
template = cv2.imread(watermark_template_path, 0)
if template is None:
    raise FileNotFoundError(f"Error: Unable to read the watermark template from {watermark_template_path}")

w, h = template.shape[::-1]

# Function to remove watermark using template matching
def remove_watermark(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply template matching to locate the watermark
    result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Check if the maximum value is high enough to consider a match
    threshold = 0.8  # Adjust this value as needed
    if max_val >= threshold:
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        # Create a mask over the detected watermark
        mask = np.zeros_like(image[:, :, 0])
        mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255
        
        # Inpaint to remove the watermark
        inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        return inpainted_image
    else:
        print("Warning: Watermark not detected with sufficient confidence.")
        return image

# Process each image
for file_path in image_files:
    # Read the image
    image = cv2.imread(file_path)
    
    # Remove the watermark
    result_image = remove_watermark(image)
    
    # Save the result in the output folder with the same file name
    output_path = os.path.join(output_folder, os.path.basename(file_path))
    cv2.imwrite(output_path, result_image)

print("Watermark removal completed. Check the OUTPUT folder.")
