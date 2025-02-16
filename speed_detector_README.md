# Vehicle Speed Detection System

This system provides real-time vehicle speed detection and monitoring using computer vision techniques.

## Features

- Real-time vehicle detection and tracking
- Speed calculation in MPH/KPH
- Speed limit violation detection
- Visual speed display
- Support for multiple vehicles
- Speed violation logging

## Requirements

See `speed_detector_requirements.txt` for detailed dependencies.

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r speed_detector_requirements.txt
```

## Usage

1. Run the main script:
```bash
python traffic_violation_detector.py
```

2. The system will:
   - Access your camera feed
   - Detect and track vehicles
   - Calculate and display vehicle speeds
   - Show warnings for speed violations

3. Press 'q' to quit the application

## Configuration

You can modify detection parameters in the `SpeedDetector` class:
- Speed limit threshold
- Detection confidence threshold
- Pixels per meter calibration
- Frame processing resolution

## Calibration

For accurate speed measurements:
1. Set the `pixels_per_meter` value based on your camera setup
2. Adjust detection area based on camera position
3. Calibrate speed calculations using known reference speeds

## Troubleshooting

1. Inaccurate speed readings:
   - Recalibrate pixels_per_meter value
   - Verify camera position and angle
   - Check frame rate consistency

2. Poor detection:
   - Adjust lighting conditions
   - Modify detection thresholds
   - Check camera resolution

## License

MIT License
