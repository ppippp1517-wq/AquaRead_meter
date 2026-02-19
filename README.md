# AquaRead: Automatic Analog Water Meter Reading System

AquaRead is an Automatic Meter Reading (AMR) system designed to read analog water meters using computer vision and deep learning techniques. The system integrates multiple AI modules in a modular end-to-end pipeline to reduce human error in manual meter reading and improve operational efficiency.

---

## Overview

Manual water meter reading is prone to human error, inconsistencies, and operational costs. AquaRead addresses these issues by applying deep learning-based image processing techniques to automatically interpret analog water meter readings captured in real-world environments.

The system is designed in a modular architecture, allowing each component to be developed, evaluated, and improved independently.

---

## System Architecture Diagram

![AquaRead System Architecture](system_architecture.png)

---

## System Pipeline (End-to-End)

The AquaRead system operates through the following pipeline:

1. **Image Acquisition**  
   Capture images of analog water meters using mobile or handheld devices.

2. **ROI Detection (YOLO-based)**  
   Detect and localize key regions on the meter face, including:
   - Digital roller digits  
   - Analog pointer dials  

3. **Digital Reading Module**  
   Recognize roller-type digits using a CNN-based digit classification model.

4. **Pointer Estimation Module**  
   Estimate analog pointer values through angle regression and scale mapping.

5. **Sanity Validation and Fusion**  
   Validate and fuse digital and analog readings to produce a final, reliable water consumption value.

---

## Module Description

### 1) YOLO-based ROI Detection
A YOLOv8-based object detection model is used to identify relevant regions on the water meter face, enabling robust localization under varying lighting and background conditions.

### 2) Digital Reader Module
Detected digit regions are cropped and processed by a CNN classifier to recognize individual digits, forming the digital meter reading.

### 3) Pointer Estimation Module
Analog pointer dials are processed by a regression-based model that estimates pointer angles and converts them into numerical values.

### 4) Sanity Validation and Fusion
A rule-based validation mechanism combines digital and analog outputs, correcting inconsistencies and ensuring temporal continuity with previous readings.

---

## System Implementation

- **Backend:** Python, Django  
- **AI Models:** YOLOv8, CNN, Regression-based models  
- **Image Processing:** OpenCV, NumPy  
- **Frontend:** Web Application and Mobile WebView  
- **Database:** SQLite (Prototype)

---

## Experimental Results

Experimental evaluations demonstrate that the AquaRead system can accurately read analog water meters under real-world conditions. The system shows robustness against moderate variations in lighting, orientation, and meter design. Detailed quantitative results and evaluation metrics are provided in the accompanying thesis document.

---

## Limitations

- The system has been trained and evaluated on a limited variety of water meter designs.  
- Performance may degrade when applied to meter types with significantly different layouts or pointer configurations.  
- Severe occlusion or extreme lighting conditions may affect detection accuracy.  

---

## Future Work

- Expand the training dataset to include a wider range of meter types and manufacturers.  
- Improve robustness under challenging environmental conditions.  
- Integrate temporal sequence modeling to enhance reading consistency over time.  
- Deploy the system on edge devices for real-time field operation.  

---

## Repository Structure

```
AquaRead_meter/
│
├── README.md
├── requirements.txt
├── system_architecture.png
│
├── watermeter_project/              # Main Django project configuration
│   ├── settings.py                  # Project settings
│   ├── urls.py                      # Root URL configuration
│   └── ...
│
├── water_meter_alignment/           # Image alignment and preprocessing module
│
└── watermeter_project/meter_reader/ # Core application module
    │
    ├── views.py                     # Main system pipeline logic
    ├── models.py                    # Database models
    ├── urls.py                      # Application-level routing
    ├── camera.py                    # Image acquisition handling
    ├── utils.py                     # Utility and helper functions
    │
    ├── needle1.py                   # Analog pointer estimation logic
    ├── modelneedle.h5               # Trained pointer regression model
    │
    ├── templates/                   # HTML templates (frontend UI)
    ├── static/                      # CSS, JavaScript, and static assets
    ├── migrations/                  # Django migration files
    │
    ├── admin.py
    ├── apps.py
    ├── tests.py
    └── __init__.py
```

---

## Description

The repository follows a modular Django-based architecture.  
The `meter_reader` application serves as the core intelligent processing unit, integrating image acquisition, detection, digit recognition, pointer estimation, and validation logic within a unified web-based system.

The structure reflects the system’s layered design, separating configuration, preprocessing, AI inference modules, and frontend components to ensure maintainability and extensibility.
