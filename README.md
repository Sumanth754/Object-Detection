# Real-Time Object Detection for Retail Analytics

This project is a Python application that performs real-time object detection on a video file using the YOLOv3 model. It is designed to simulate a retail analytics scenario, such as counting customers or monitoring products.

---

## **IMPORTANT SETUP: Download Required Files**

This repository does **not** include the large model weights and video files required to run the project. You must download them manually.

**1. Download the Model Weights:**
   * **Link:** [https://pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
   * **Action:** After downloading, move this file into the `dnn_model` folder. The final path should be `dnn_model/yolov3.weights`.

**2. Download the Sample Video:**
   * **Link:** [https://www.pexels.com/video/people-inside-a-shopping-mall-6806410/download/](https://www.pexels.com/video/people-inside-a-shopping-mall-6806410/download/)
   * **Action:** After downloading, move this file into the main project folder (`RealTimeRetailAnalytics`) and **rename it to `sample.mp4`**.

Once both files are in their correct locations, you can proceed with the steps below.

---

## How to Run the Application

### Prerequisites

*   Python 3.x installed on your system.

### Step 1: Navigate to the Project Directory

Open your terminal or command prompt and navigate to the project folder:
```bash
cd C:\Users\sumanth\Desktop\RealTimeRetailAnalytics
```

### Step 2: Create a Virtual Environment (Recommended)

It is highly recommended to create a virtual environment to keep the project dependencies isolated.
```bash
python -m venv venv
```
Activate the virtual environment:
*   **On Windows:**
    ```bash
    .\venv\Scripts\activate
    ```

### Step 3: Install the Required Libraries

Install all the required Python libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

Execute the main Python script to start the object detection:
```bash
python main.py
```
An OpenCV window will open and display the video with detected objects highlighted. To stop the application, press the **'q'** key.
