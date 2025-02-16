SVVDS (Smart Vehicle Violation Detection System)

This is a comprehensive traffic monitoring solution that combines multiple detection systems to monitor and analyze traffic behavior.

1. License Plate Detector (new_license_plate_detector.py)
----------------------------------------------------
This module provides real-time license plate detection and recognition capabilities.

Key Features:
- Uses OpenCV and Tesseract OCR for plate recognition
- Includes image preprocessing for better accuracy
- Handles skewed and rotated license plates
- Supports both live webcam and video file input
- Real-time processing and display
- Error handling for missing Tesseract installation

Main Components:
- Plate detection using contour analysis
- Image enhancement using bilateral filtering
- Perspective transformation for skewed plates
- OCR text extraction and cleaning
- Real-time display with detected plate overlay

2. Traffic Violation Detector (traffic_violation_detector.py)
-------------------------------------------------------
This module detects and logs various traffic violations in real-time.

Key Features:
- Vehicle detection and tracking
- Speed monitoring
- Unsafe overtaking detection
- Lane position tracking
- Violation logging system

Components:
- Vehicle detection using Haar Cascade
- Multi-vehicle tracking system
- Speed calculation between detection zones
- Unsafe distance monitoring
- Timestamp-based violation logging

3. Speed Detection Module (Part of traffic_violation_detector.py)
-----------------------------------------------------------
Specialized module for vehicle speed detection.

Features:
- Real-time speed calculation
- Configurable speed limits
- Distance calibration system
- FPS-based measurements
- Support for multiple vehicle tracking
- Speed violation logging

4. Lane Change and Overtaking Detection
-----------------------------------
System for monitoring vehicle positions and detecting unsafe maneuvers.

Features:
- Lane position tracking
- Unsafe overtaking detection
- Distance threshold monitoring
- Real-time safety analysis
- Violation logging with timestamps

Technical Requirements:
- Python 3.7 or higher
- OpenCV 4.5.0+
- NumPy 1.19.0+
- Tesseract OCR
- dlib 19.24.0+
- SciPy 1.7.0+
- imutils 0.5.4+
- Webcam or video input source

The system is designed for traffic monitoring and enforcement purposes, providing real-time detection and logging of various traffic violations including speeding, unsafe overtaking, and improper lane changes, while also capturing license plate information for vehicle identification.
