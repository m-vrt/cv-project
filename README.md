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

### **Using the Provided Video Files**  
This repository **already includes** the required input video and pre-generated outputs for different models, so you don’t need to provide your own video.  

- **Input Video:** `for cv-project.mp4`  
- **Pre-generated Output Videos:**  
  - `output_yolo_seg.mp4` (YOLOv8-Seg)  
  - `output_faster_rcnn.mp4` (Faster R-CNN)  
  - `output_retinanet.mp4` (RetinaNet)  
  - `output_sam.mp4` (Segment Anything Model)  

### **Customizing Start Time & Output Name**  
- The **YOLOv8-Seg output (`output_yolo_seg.mp4`) starts processing from 1:00 minute by default.**  
- Modify `start_time_seconds = 60` in `yolov8.py` to change the starting point.  
- Change `output_path = "output_yolo_seg.mp4"` if you want a different output filename.  

## GitHub Repository  

The GitHub repository for this project can be found at [`m-vrt/cv-project`](https://github.com/m-vrt/cv-project).
