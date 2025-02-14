# Recognizing Objects in Video Sequences  

This repository contains the implementation of real-time object detection and segmentation using YOLOv8. It supplements the **Project Report**, which serves as the final requirement for the course **Project: Computer Vision**.

## Installation  

Clone the repository:  
```bash
git clone https://github.com/m-vrt/cv-project.git
```

Navigate to the project directory:  
```bash
cd cv-project
```

Set up a virtual environment:  
```bash
python3 -m venv venv
```

Activate the virtual environment:  

**On Windows:**  
```bash
venv\Scripts\activate
```

**On macOS and Linux:**  
```bash
source venv/bin/activate
```

Install project dependencies:  
```bash
pip install -r requirements.txt
```

## Development Environment  

This project was developed using **Visual Studio Code (VS Code)**.

## Requirements  

- **Python** == 3.10.0  
- **OpenCV** == 4.11.0.86  
- **PyTorch** == 2.6.0+cu118  
- **NumPy** == 1.26.4  
- **Ultralytics** == 8.3.75  

## Usage  

Run the following command to process your video and generate an output:
```bash
python yolov8.py
```
Ensure that your input video is named `for cv-project.mp4` and placed in the same folder as the script. The processed output will be saved as `output_yolo_seg.mp4` in the same directory.

### Customizing Start Time & Output Name
- This script starts processing from **1:00 minute** by default.  
- Modify `start_time_seconds = 60` in `yolov8.py` to start at a different time.
- Change `output_path = "output_yolo_seg.mp4"` if you want a different output filename.

## GitHub Repository  

The GitHub repository for this project can be found at:  
[https://github.com/m-vrt/cv-project](https://github.com/m-vrt/cv-project)
