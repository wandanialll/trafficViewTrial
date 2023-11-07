import cv2
import numpy as np
import os
import time

# Step 1: Install necessary dependencies (if not already installed)
# pip install opencv-python
# pip install opencv-python-headless
# pip install requests

# Step 2: Download YOLOv3 weights and configuration files from the official YOLO website
#yolo_weights_url = "https://pjreddie.com/media/files/yolov3.weights"
#yolo_cfg_url = "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg"
#yolo_names_url = "https://github.com/pjreddie/darknet/blob/master/data/coco.names"

weights_file = "traffic\\yolov3.weights"
cfg_file = "traffic\\yolov3.cfg"
names_file = "coco.names"

# Step 3: Load the YOLOv3 model using OpenCV's DNN module
net = cv2.dnn.readNet(weights_file, cfg_file)

# Load COCO class names
with open(names_file, "r") as f:
    classes = f.read().strip().split("\n")

# Step 4: Process input images and detect vehicles
def detect_vehicles(image_path, scale_factor=1.0):
    image = cv2.imread(image_path)
    
    # Scale up the image if the scale_factor is greater than 1.0
    if scale_factor > 1.0:
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

    height, width = image.shape[:2]
    image_area = height * width

    # Create a blob from the image and set it as input to the neural network
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (608, 608), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    out_layer_names = net.getUnconnectedOutLayersNames()

    # Forward pass to get the predictions
    detections = net.forward(out_layer_names)

    box_count = 0  # Counter for the number of detected boxes

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Get the bounding box coordinates
            center_x, center_y, w, h = (obj[:4] * np.array([width, height, width, height])).astype(int)
            x, y = center_x - w // 2, center_y - h // 2

            # Calculate the area of the detected object
            object_area = w * h

            # Filter detections for road vehicles (COCO class IDs 2, 3, 5, 7) and size less than 20% of the image
            if class_id in [2, 3, 5, 7] and confidence > 0.45 and object_area < 0.2 * image_area:
                box_count += 1

                # Draw bounding box and label on the image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return image, box_count

# Step 5: Save the processed images with identified vehicles to the "processed_Gua_Tempurung" folder
# Step 5: Save the processed images with identified vehicles and append to the tabbed text file
if not os.path.exists("traffic\processed_Gua_Tempurung"):
    os.makedirs("traffic\processed_Gua_Tempurung")

# Tabbed text file path to store the time and number of boxes
output_txt_file = os.path.join("traffic", "processed_Gua_Tempurung", "box_count.txt")

# Process all images in the "input_images" folder and save them in the "processed_Gua_Tempurung" folder
input_images_folder = os.path.join("traffic", "gua tempurung")
image_files = [f for f in os.listdir(input_images_folder) if os.path.isfile(os.path.join(input_images_folder, f))]

# Sort the image files based on modification time (latest first)
image_files.sort(key=lambda x: os.path.getmtime(os.path.join(input_images_folder, x)), reverse=True)

# Process the latest image only
if image_files:
    latest_image = image_files[0]
    input_image_path = os.path.join(input_images_folder, latest_image)
    scale_factor = 2.0  # Scale up the image by 1.5 times
    output_image, box_count = detect_vehicles(input_image_path, scale_factor)
    # Save processed image
    output_image_path = os.path.join("traffic", "processed_Gua_Tempurung", latest_image)
    cv2.imwrite(output_image_path, output_image)

    # Append timestamp and box count to the text file
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(output_txt_file, "a") as txt_output:
        txt_output.write(f"{timestamp}\t{box_count}\n")

    print("Latest image processed and saved to 'processed_Gua_Tempurung' folder.")
    print("Timestamp and box count appended to 'box_count.txt'.")
else:
    print("No images found in the 'gua tempurung' folder.")