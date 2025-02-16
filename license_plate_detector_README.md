# License Plate Detection System

This system provides real-time license plate detection and recognition using computer vision techniques.

## Features

- Real-time license plate detection from video feed
- OCR-based plate number recognition
- Support for multiple camera sources
- High-accuracy detection using deep learning models
- Real-time visualization of detected plates

## Requirements

See `license_plate_detector_requirements.txt` for detailed dependencies.

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r license_plate_detector_requirements.txt
```

## Usage

1. Run the main script:
```bash
python new_license_plate_detector.py
```

2. The system will:
   - Access your camera feed
   - Detect license plates in real-time
   - Display the video feed with detected plates
   - Show recognized plate numbers

3. Press 'q' to quit the application

## Configuration

You can modify detection parameters in the `LicensePlateDetector` class:
- Detection confidence threshold
- OCR confidence threshold
- Frame processing resolution

## Troubleshooting

1. If camera doesn't open:
   - Check camera permissions
   - Verify camera index (default is 0)

2. If detection is slow:
   - Lower the processing resolution
   - Reduce frame processing frequency

## License

MIT License
