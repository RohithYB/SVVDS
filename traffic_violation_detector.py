import cv2
import numpy as np
import dlib
import time
from scipy.spatial import distance as dist
import imutils

class TrafficViolationDetector:
    def __init__(self):
        # Initialize vehicle detector
        self.car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
        
        # Parameters for speed calculation
        self.distance = 20  # meters between detection lines
        self.speed_limit = 60  # km/h
        
        # Vehicle tracking parameters
        self.tracked_vehicles = {}  # Dictionary to store vehicle tracking info
        self.next_vehicle_id = 1
        self.detection_lines = []  # List to store detection lines
        self.min_detection_confidence = 0.6
        
        # Parameters for overtaking detection
        self.lane_positions = []  # Will store lane markings
        self.unsafe_overtaking_threshold = 2  # meters (minimum safe distance)
        self.overtaking_history = defaultdict(list)
        
        # Violation logging
        self.violations = []
        
        # Speed detection parameters
        self.speed_detector = SpeedDetector(speed_limit_mph=30)
        
    def setup_detection_zones(self, frame_height, frame_width):
        """Setup detection lines and lanes for the given frame dimensions."""
        # Create two detection lines for speed measurement
        line1_y = int(frame_height * 0.3)  # First line at 30% of frame height
        line2_y = int(frame_height * 0.7)  # Second line at 70% of frame height
        self.detection_lines = [(0, line1_y, frame_width, line1_y),
                              (0, line2_y, frame_width, line2_y)]
        
        # Setup lane positions (assuming 3 lanes)
        lane_width = frame_width // 3
        self.lane_positions = [lane_width, lane_width * 2]
    
    def detect_vehicles(self, frame):
        """Detect vehicles in the frame using Haar cascade classifier."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vehicles = self.car_cascade.detectMultiScale(gray, 1.1, 3)
        return vehicles
    
    def calculate_speed(self, vehicle_id, y_pos, timestamp):
        """Calculate vehicle speed based on time between detection lines."""
        if vehicle_id in self.tracked_vehicles:
            vehicle_data = self.tracked_vehicles[vehicle_id]
            if 'first_detection' in vehicle_data:
                # Calculate time difference
                time_diff = timestamp - vehicle_data['first_detection']['time']
                if time_diff > 0:
                    # Calculate speed (km/h)
                    distance = abs(y_pos - vehicle_data['first_detection']['y'])
                    pixels_to_meters = self.distance / abs(self.detection_lines[1][1] - self.detection_lines[0][1])
                    real_distance = distance * pixels_to_meters
                    speed = (real_distance / time_diff) * 3.6  # Convert m/s to km/h
                    return speed
        return None
    
    def detect_unsafe_overtaking(self, vehicle_positions):
        """Detect unsafe overtaking maneuvers."""
        unsafe_overtaking = []
        
        # Sort vehicles by x position (horizontal position on road)
        sorted_vehicles = sorted(vehicle_positions, key=lambda x: x[1])
        
        # Check distances between adjacent vehicles
        for i in range(len(sorted_vehicles)-1):
            curr_vehicle = sorted_vehicles[i]
            next_vehicle = sorted_vehicles[i+1]
            
            # Calculate horizontal distance between vehicles
            distance = abs(next_vehicle[1] - curr_vehicle[1])
            
            # Check if distance is less than safety threshold
            if distance < self.unsafe_overtaking_threshold:
                unsafe_overtaking.append((curr_vehicle[0], next_vehicle[0]))
        
        return unsafe_overtaking
    
    def log_violation(self, violation_type, vehicle_id, speed=None):
        """Log traffic violations with timestamp."""
        timestamp = datetime.datetime.now()
        violation = {
            'timestamp': timestamp,
            'type': violation_type,
            'vehicle_id': vehicle_id,
            'speed': speed
        }
        self.violations.append(violation)
        print(f"Violation detected: {violation_type} by vehicle {vehicle_id}" +
              (f" at speed {speed:.1f} km/h" if speed else ""))
    
    def process_frame(self, frame):
        """Process a single frame for violation detection."""
        if not self.detection_lines:
            self.setup_detection_zones(frame.shape[0], frame.shape[1])
        
        # Detect vehicles
        vehicles = self.detect_vehicles(frame)
        current_timestamp = time.time()
        vehicle_positions = []
        
        # Process each detected vehicle
        for (x, y, w, h) in vehicles:
            vehicle_center_x = x + w//2
            vehicle_center_y = y + h//2
            
            # Check if vehicle crosses detection lines
            for line_y in [self.detection_lines[0][1], self.detection_lines[1][1]]:
                if abs(vehicle_center_y - line_y) < 10:  # Vehicle is crossing a detection line
                    # Try to match with existing vehicle or create new tracking
                    matched = False
                    for v_id, v_data in self.tracked_vehicles.items():
                        if abs(vehicle_center_x - v_data.get('last_x', 0)) < 50:  # Assuming 50 pixels tolerance
                            matched = True
                            if 'first_detection' not in v_data:
                                v_data['first_detection'] = {'time': current_timestamp, 'y': vehicle_center_y}
                            else:
                                # Calculate speed
                                speed = self.calculate_speed(v_id, vehicle_center_y, current_timestamp)
                                if speed and speed > self.speed_limit:
                                    self.log_violation('SPEEDING', v_id, speed)
                            
                            v_data['last_x'] = vehicle_center_x
                            v_data['last_y'] = vehicle_center_y
                            v_data['last_seen'] = current_timestamp
                            break
                    
                    if not matched:
                        # Create new vehicle tracking
                        self.tracked_vehicles[self.next_vehicle_id] = {
                            'first_detection': {'time': current_timestamp, 'y': vehicle_center_y},
                            'last_x': vehicle_center_x,
                            'last_y': vehicle_center_y,
                            'last_seen': current_timestamp
                        }
                        self.next_vehicle_id += 1
            
            # Store vehicle position for overtaking detection
            vehicle_positions.append((self.next_vehicle_id-1, vehicle_center_x, vehicle_center_y))
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Detect unsafe overtaking
        if len(vehicle_positions) >= 2:
            unsafe_overtaking = self.detect_unsafe_overtaking([(vid, x) for vid, x, y in vehicle_positions])
            for v1_id, v2_id in unsafe_overtaking:
                self.log_violation('UNSAFE_OVERTAKING', f'{v1_id},{v2_id}')
        
        # Draw detection lines
        for line in self.detection_lines:
            cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)
        
        # Draw lane markings
        for x in self.lane_positions:
            cv2.line(frame, (x, 0), (x, frame.shape[0]), (255, 255, 0), 2)
        
        # Process frame for speed detection
        speed_frame = self.speed_detector.process_frame(frame)
        
        return speed_frame
    
    def start_detection(self):
        """Start real-time traffic violation detection."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not access webcam")
            return
        
        # Set lower resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Starting traffic violation detection. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display violation counts
                speed_violations = sum(1 for v in self.violations if v['type'] == 'SPEEDING')
                overtaking_violations = sum(1 for v in self.violations if v['type'] == 'UNSAFE_OVERTAKING')
                
                cv2.putText(processed_frame, f"Speed Violations: {speed_violations}",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(processed_frame, f"Unsafe Overtaking: {overtaking_violations}",
                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Show frame
                cv2.imshow("Traffic Violation Detection", processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            cap.release()
            cv2.destroyAllWindows()

class SpeedDetector:
    def __init__(self, speed_limit_mph=30):
        self.speed_limit = speed_limit_mph
        self.tracker = dlib.correlation_tracker()
        self.tracking = False
        self.fps = 30  # Default FPS
        # Real-world distance (in meters) per pixel - needs calibration
        self.pixels_per_meter = 0.1
        self.current_position = None
        self.prev_position = None
        self.prev_time = None
        
    def calculate_speed(self, pixels_moved, time_diff):
        """Calculate speed in MPH given pixels moved and time difference"""
        if time_diff == 0:
            return 0
        
        # Convert pixels to meters
        meters = pixels_moved * self.pixels_per_meter
        
        # Calculate speed in meters per second
        speed_mps = meters / time_diff
        
        # Convert to MPH
        speed_mph = speed_mps * 2.237
        
        return speed_mph
    
    def process_frame(self, frame):
        """Process a single frame and detect speed"""
        frame = imutils.resize(frame, width=800)
        
        if not self.tracking:
            # Detect vehicles using HOG detector
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # Detect vehicles in the frame
            boxes, _ = hog.detectMultiScale(frame, winStride=(4, 4),
                                          padding=(8, 8), scale=1.05)
            
            if len(boxes) > 0:
                # Start tracking the first detected vehicle
                box = boxes[0]
                (x, y, w, h) = box
                rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                self.tracker.start_track(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), rect)
                self.tracking = True
                self.current_position = (x + w//2, y + h//2)
                self.prev_position = self.current_position
                self.prev_time = time.time()
        
        else:
            # Update the tracker
            self.tracker.update(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pos = self.tracker.get_position()
            
            # Get current position
            self.current_position = (int(pos.left() + pos.width()//2),
                                   int(pos.top() + pos.height()//2))
            
            current_time = time.time()
            
            if self.prev_position and self.prev_time:
                # Calculate pixels moved
                pixels_moved = dist.euclidean(self.prev_position, self.current_position)
                time_diff = current_time - self.prev_time
                
                # Calculate speed
                speed = self.calculate_speed(pixels_moved, time_diff)
                
                # Draw tracking box and speed
                cv2.rectangle(frame, (int(pos.left()), int(pos.top())),
                            (int(pos.right()), int(pos.bottom())),
                            (0, 255, 0), 2)
                
                speed_text = f"Speed: {speed:.2f} MPH"
                cv2.putText(frame, speed_text,
                          (int(pos.left()), int(pos.top() - 10)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Check for speed limit violation
                if speed > self.speed_limit:
                    cv2.putText(frame, "SPEED VIOLATION!",
                              (int(pos.left()), int(pos.bottom() + 20)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            self.prev_position = self.current_position
            self.prev_time = current_time
            
        return frame

def main():
    detector = TrafficViolationDetector()
    detector.start_detection()

if __name__ == "__main__":
    main()
