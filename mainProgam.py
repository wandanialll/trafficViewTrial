from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
import base64
from PIL import Image
from io import BytesIO
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import cv2
import numpy as np
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
names_file = "traffic\\coco.names"

# Step 3: Load the YOLOv3 model using OpenCV's DNN module
net = cv2.dnn.readNet(weights_file, cfg_file)

# Load COCO class names
with open(names_file, "r") as f:
    classes = f.read().strip().split("\n")

# Function to download images using Selenium
def download_images():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)  # Change to webdriver.Firefox() if using Firefox

    url = "https://www.llm.gov.my/awam/cctv"  # Replace with the URL of the webpage you want to visit
    driver.get(url)

    dropdown_menu = Select(driver.find_element(By.ID, "cctvhighway"))
    desired_option = "E36:Jambatan Pulau Pinang (PNB)"  # Replace this with the option you want to select
    dropdown_menu.select_by_visible_text(desired_option)

    wait = WebDriverWait(driver, 10)
    wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "jconfirm-content-pane")))

    container = driver.find_element(By.CLASS_NAME, "jconfirm-content-pane")
    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "img.border")))
    container_source = container.get_attribute("innerHTML")
    soup = BeautifulSoup(container_source, "html.parser")
    images = soup.find_all("img", {"class": "border"})

    folder_name = "traffic/gua tempurung"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    desired_title = "PNB CAM 11 PNG BRG KM5.40 WB"
    file_to_delete = f"{folder_name}/{desired_title}.png"
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)

    for idx, image in enumerate(images):
        image_title = image.get("title")
        if image_title == desired_title:
            image_base64_data = image.get("src").split(",")[-1]  # Extract the Base64 data part
            image_data = base64.b64decode(image_base64_data)

            file_name = f"{folder_name}/{image_title}_{timestamp}.png"
            if os.path.exists(file_name):
                os.remove(file_name)

            with open(file_name, "wb") as f:
                f.write(image_data)

            print(f"Image with the title '{desired_title}' downloaded and saved successfully.")
            break

    else:
        print(f"Image with the title '{desired_title}' not found.")

    driver.quit()

# Function to process input images and detect vehicles

def detect_vehicles(image_path, scale_factor=2.0):
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

    boxes = []
    confidences = []

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
                # Store the bounding box and confidence for NMS
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Apply Non-Maximum Suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.45, nms_threshold=0.5)

    box_count = 0  # Counter for the number of detected boxes

    # Convert the indices to a simple list
    indices = np.squeeze(indices)

    for i in indices:
        x, y, w, h = boxes[i]

        # Draw bounding box and label on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{classes[class_id]}: {confidences[i]:.2f}"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        box_count += 1

    return image, box_count

# Function to loop the entire process every 2 minutes
processed_images = set()

def main_loop():
    while True:
        download_images()
        input_images_folder = os.path.join("traffic", "gua tempurung")
        with open(output_txt_file, "a") as txt_output:
            for image_filename in os.listdir(input_images_folder):
                if image_filename not in processed_images:
                    input_image_path = os.path.join(input_images_folder, image_filename)
                    scale_factor = 2.0  # Scale up the image by 1.5 times
                    output_image, box_count = detect_vehicles(input_image_path)

                    output_image_path = os.path.join("traffic", "processed_Gua_Tempurung", image_filename)
                    cv2.imwrite(output_image_path, output_image)

                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    txt_output.write(f"{timestamp}\t{box_count}\n")

                    # Add the processed image filename to the set
                    processed_images.add(image_filename)

        print("Vehicle detection completed and processed images saved to 'processed_Gua_Tempurung' folder.")
        print("Timestamps and box counts appended to 'box_count.txt'.")
        time.sleep(120)  # Wait for 2 minutes before running the loop again

if __name__ == "__main__":
    if not os.path.exists("traffic/processed_Gua_Tempurung"):
        os.makedirs("traffic/processed_Gua_Tempurung")

    output_txt_file = os.path.join("traffic", "processed_Gua_Tempurung", "box_count.txt")
    main_loop()