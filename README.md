# trafficViewTrial
 An ongoing personal project using object detection, python and in the future a model to calculate and predict traffic

 Purpose of the project was to try out object detection and integrate it with Mathematic principles such as:
 i. Regression Analysis
 ii. Time Series Analysis
 iii. Probability and Statistics

 In the current form, the project is limited to object detection and tracking of number of objects. In the future, the numerical data obtained from the object detection algorithm will be calculated to enable traffic prediction and share current traffic conditions.

 How the program works:

  1. A webscrapping program (datasource.py/imageScrape64.py) obtains the photo from Lembaga Lebuhraya Malaysia website and stores at a fixed interval.
  2. The driver program (mainProgram) then runs the object detection process to identify objects in the current image.
  2.1. Currently there is a problem whereby the tags are incorrect, tagging vehicles as person
  2.2. The program also keeps count of how many objects are detected in the current image
  3. The processed image is kept in a folder for the sake of review and debugging.
  4. In the future, data is going to be calculated and tabularised.

 Dependencies:

 1. Object detection model. Depending on the system, models used are:
    i. YoloV4
    ii. YoloV3
    iii. YoloV3-tiny
    iv. YoloV2-tiny
    v. mobilenet-ssd
 2. For faster processing, an NVDIA GPU is needed for CUDA support.
    
 Appendix:

 ![image](https://github.com/wandanialll/trafficViewTrial/assets/123443949/c4740ec5-4365-4aac-b7bb-8609d0ac4088)
 Raw image obtain from LLM website before upscalling and processing

 ![image](https://github.com/wandanialll/trafficViewTrial/assets/123443949/ef930bd1-e26f-41de-85e1-125250468557)
 Image that has been processed and object has been identified

 Data of number of vehicles and timestamp that is kept for future calculations
 [Date]     [Time]   [No. of Vehicles]
 2023-07-27 16:33:47	8
 2023-07-27 16:35:59	16
 2023-07-27 16:38:12	4
 2023-07-27 16:40:27	10
 2023-07-27 16:42:40	12
 2023-07-27 16:44:53	14
 2023-07-27 16:47:06	10
 2023-07-27 16:49:18	17
 2023-07-27 16:51:35	7
