# Overtaking Safety Prediction System

This system predicts safe overtaking opportunities and potential collision risks in real-time using computer vision.

## Features

- Real-time vehicle detection and tracking
- Collision risk assessment
- Safe overtaking prediction
- Time-to-collision (TTC) calculation
- Visual safety alerts
- Blind spot monitoring
- Future path prediction

## Requirements

See `overtaking_predictor_requirements.txt` for detailed dependencies.

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r overtaking_predictor_requirements.txt
```

## Usage

1. Run the main script:
```bash
python overtaking_predictor.py
```

2. The system will:
   - Access your camera feed
   - Track surrounding vehicles
   - Calculate collision risks
   - Predict safe overtaking opportunities
   - Display visual safety information

3. Press 'q' to quit the application

## Safety Parameters

You can customize safety thresholds in the `OvertakingPredictor` class:
```python
self.safe_distance_front = 150  # Minimum safe distance in front
self.safe_distance_side = 100   # Minimum safe distance to side
self.time_to_collision_threshold = 3.0  # Seconds
self.prediction_time = 2.0  # Seconds to look ahead
```

## Visual Indicators

- Green box: User's vehicle
- Red box: Other vehicles
- Red circle: High collision risk
- Predicted path lines
- Safety status messages
- Risk level indicators

## Safety Features

1. Collision Risk Assessment:
   - Time to collision calculation
   - Relative speed monitoring
   - Distance tracking
   - Risk level classification

2. Overtaking Safety:
   - Blind spot detection
   - Safe distance monitoring
   - Speed differential analysis
   - Future position prediction

## Troubleshooting

1. False collision warnings:
   - Adjust safety distance thresholds
   - Modify time to collision threshold
   - Check vehicle detection sensitivity

2. Missed vehicle detection:
   - Verify camera position
   - Adjust detection parameters
   - Check lighting conditions

## Important Note

This system is designed as a driving assistance tool only. Always rely on your own judgment and follow traffic rules and regulations.

## License

MIT License
