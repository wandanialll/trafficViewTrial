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

#webdriver options
chrome_options = Options()
chrome_options.add_argument("--headless")
# Set up the WebDriver for your preferred browser
driver = webdriver.Chrome(options=chrome_options)  # Change to webdriver.Firefox() if using Firefox

# Navigate to the webpage
url = "https://www.llm.gov.my/awam/cctv"  # Replace with the URL of the webpage you want to visit
driver.get(url)

# Assuming there's a drop-down menu with the id "cctvhighway" and you want to select "PLS"
dropdown_menu = Select(driver.find_element(By.ID, "cctvhighway"))

# Choose the desired option using select_by_visible_text
desired_option = "E1:L/raya Utara Selatan (PLUS Utara)"  # Replace this with the option you want to select
dropdown_menu.select_by_visible_text(desired_option)

# Wait for the container with the images to be visible after making the selection
wait = WebDriverWait(driver, 10)
wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "jconfirm-content-pane")))

# Find the container with the images using its class name
container = driver.find_element(By.CLASS_NAME, "jconfirm-content-pane")

# Wait for the images to load within the container (adjust the timeout as needed)
wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "img.border")))

# Get the page source of the container
container_source = container.get_attribute("innerHTML")

# Use BeautifulSoup to parse the HTML in the container and find all images with the class "border"
soup = BeautifulSoup(container_source, "html.parser")
images = soup.find_all("img", {"class": "border"})

# Create a folder to save the images
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#folder_name = f"downloaded_images_{timestamp}"
#os.makedirs(folder_name, exist_ok=True)
folder_name = "traffic\gua tempurung"

# Download the specific image with the desired title and save in the folder
desired_title = "PLS CAM C1 GUA TEMPURUNG KM305.4 NB"  # Replace this with the title of the image you want to download
for idx, image in enumerate(images):
    image_title = image.get("title")
    if image_title == desired_title:
        image_base64_data = image.get("src").split(",")[-1]  # Extract the Base64 data part
        
        # Convert the Base64 data to image data
        image_data = base64.b64decode(image_base64_data)
        
        # Save the image to a file with a timestamp in the title
        file_name = f"{folder_name}/{image_title}_{timestamp}.png"
        with open(file_name, "wb") as f:
            f.write(image_data)
            
        print(f"Image with the title '{desired_title}' downloaded and saved successfully.")
        break  # Stop searching once the image is found

else:
    print(f"Image with the title '{desired_title}' not found.")

# Close the browser
driver.quit()
