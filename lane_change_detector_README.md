# Lane Change Detection System

This system detects and analyzes vehicle lane changes using computer vision techniques.

## Features

- Real-time lane line detection
- Vehicle tracking across lanes
- Lane change event detection
- Visual lane marking
- Multiple vehicle tracking
- Lane change history logging

## Requirements

See `lane_change_detector_requirements.txt` for detailed dependencies.

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r lane_change_detector_requirements.txt
```

## Usage

1. Run the main script:
```bash
python lane_change_detector.py
```

2. The system will:
   - Access your camera feed
   - Detect lane lines
   - Track vehicles
   - Identify lane changes
   - Display visual feedback

3. Press 'q' to quit the application

## Configuration

You can modify detection parameters in the `LaneChangeDetector` class:
- Lane detection parameters
- Vehicle tracking thresholds
- Detection region settings
- Visualization options

## Customization

1. Lane Detection:
   - Adjust Canny edge detection thresholds
   - Modify Hough transform parameters
   - Configure ROI vertices

2. Vehicle Tracking:
   - Adjust minimum vehicle size
   - Modify tracking parameters
   - Configure lane change cooldown time

## Troubleshooting

1. Poor lane detection:
   - Adjust lighting conditions
   - Modify edge detection thresholds
   - Check camera angle and position

2. Missed lane changes:
   - Adjust tracking sensitivity
   - Modify lane boundaries
   - Check vehicle detection thresholds

## License

MIT License
