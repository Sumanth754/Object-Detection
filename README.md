# Real-Time Object Detection for Retail Analytics

This project is a Python application that performs real-time object detection on a video file using the YOLOv3 model. It is designed to simulate a retail analytics scenario, such as counting customers or monitoring products.

## Project Structure

```
/RealTimeRetailAnalytics
|-- /dnn_model
|   |-- yolov3.weights  (Downloaded during setup)
|   |-- yolov3.cfg      (Downloaded during setup)
|   |-- coco.names      (Downloaded during setup)
|-- main.py
|-- sample.mp4          (Downloaded during setup)
|-- requirements.txt
|-- README.md
|-- .gitignore
```

## How to Run the Application

### Prerequisites

*   Python 3.x installed on your system.
*   An internet connection to download the model files on the first run.

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
*   **On macOS/Linux:**
    ```bash
    source venv/bin/activate
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

An OpenCV window will open and display the video with detected objects highlighted by bounding boxes. To stop the application, press the **'q'** key on your keyboard.
