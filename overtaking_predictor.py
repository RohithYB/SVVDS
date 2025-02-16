import cv2
import numpy as np
from collections import deque
import time
import math

class OvertakingPredictor:
    def __init__(self):
        # Parameters for vehicle tracking
        self.vehicle_history = {}  # Track vehicle positions over time
        self.history_length = 30   # Number of frames to keep in history
        self.min_detection_confidence = 0.6
        
        # Safety parameters (in pixels, will need calibration for real-world metrics)
        self.safe_distance_front = 150  # Minimum safe distance in front
        self.safe_distance_side = 100   # Minimum safe distance to side
        self.time_to_collision_threshold = 3.0  # Seconds
        self.prediction_time = 2.0  # Seconds to look ahead for predictions
        
        # Initialize vehicle detector
        self.car_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_car.xml'
        )
        
        # Initialize state variables
        self.frame_count = 0
        self.fps = 30  # Will be updated during processing
        self.last_frame_time = time.time()
        
    def detect_vehicles(self, frame):
        """Detect vehicles in the frame using Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vehicles = self.car_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        
        detected = []
        for (x, y, w, h) in vehicles:
            confidence = 1.0  # Placeholder for detection confidence
            if confidence > self.min_detection_confidence:
                detected.append({
                    'bbox': (x, y, w, h),
                    'centroid': (x + w//2, y + h//2),
                    'confidence': confidence
                })
        
        return detected
    
    def update_vehicle_tracking(self, vehicles):
        """Update vehicle tracking history."""
        current_time = time.time()
        
        # Calculate actual FPS
        if self.frame_count > 0:
            self.fps = 1.0 / (current_time - self.last_frame_time)
        self.last_frame_time = current_time
        
        # Update tracking for each detected vehicle
        current_vehicles = {}
        
        for vehicle in vehicles:
            centroid = vehicle['centroid']
            best_match = None
            min_distance = float('inf')
            
            # Find the closest match in previous frame
            for vid, hist in self.vehicle_history.items():
                if hist['positions']:
                    last_pos = hist['positions'][-1]
                    dist = np.sqrt((centroid[0] - last_pos[0])**2 + 
                                 (centroid[1] - last_pos[1])**2)
                    if dist < min_distance and dist < 100:  # Max pixel distance for matching
                        min_distance = dist
                        best_match = vid
            
            if best_match is None:
                # New vehicle
                vid = max(self.vehicle_history.keys(), default=0) + 1
                current_vehicles[vid] = {
                    'positions': deque(maxlen=self.history_length),
                    'timestamps': deque(maxlen=self.history_length),
                    'bbox': vehicle['bbox']
                }
            else:
                # Existing vehicle
                current_vehicles[best_match] = self.vehicle_history[best_match]
                
            # Update position history
            current_vehicles[best_match or vid]['positions'].append(centroid)
            current_vehicles[best_match or vid]['timestamps'].append(current_time)
            current_vehicles[best_match or vid]['bbox'] = vehicle['bbox']
        
        self.vehicle_history = current_vehicles
        self.frame_count += 1
    
    def calculate_vehicle_metrics(self, vehicle_id):
        """Calculate speed and direction for a vehicle."""
        hist = self.vehicle_history[vehicle_id]
        if len(hist['positions']) < 2:
            return None, None, None
        
        # Calculate speed and direction from last two positions
        pos1 = np.array(hist['positions'][-2])
        pos2 = np.array(hist['positions'][-1])
        t1 = hist['timestamps'][-2]
        t2 = hist['timestamps'][-1]
        
        time_diff = t2 - t1
        if time_diff == 0:
            return None, None, None
        
        displacement = pos2 - pos1
        speed = np.linalg.norm(displacement) / time_diff  # pixels per second
        direction = math.atan2(displacement[1], displacement[0])
        
        # Predict future position
        future_pos = pos2 + (displacement / time_diff) * self.prediction_time
        
        return speed, direction, future_pos
    
    def predict_collision_risk(self, user_vehicle_id):
        """Predict collision risks with other vehicles."""
        if user_vehicle_id not in self.vehicle_history:
            return []
        
        risks = []
        user_metrics = self.calculate_vehicle_metrics(user_vehicle_id)
        if user_metrics is None:
            return risks
        
        user_speed, user_direction, user_future = user_metrics
        user_pos = np.array(self.vehicle_history[user_vehicle_id]['positions'][-1])
        
        for vid, hist in self.vehicle_history.items():
            if vid == user_vehicle_id or len(hist['positions']) < 2:
                continue
            
            # Calculate other vehicle metrics
            other_metrics = self.calculate_vehicle_metrics(vid)
            if other_metrics is None:
                continue
                
            other_speed, other_direction, other_future = other_metrics
            other_pos = np.array(hist['positions'][-1])
            
            # Calculate relative position and velocity
            rel_pos = other_pos - user_pos
            rel_speed = other_speed - user_speed
            
            # Calculate time to collision (TTC)
            distance = np.linalg.norm(rel_pos)
            if rel_speed != 0:
                ttc = distance / abs(rel_speed)
            else:
                ttc = float('inf')
            
            # Check for collision risk
            if ttc < self.time_to_collision_threshold and distance < self.safe_distance_front:
                risk_level = 1.0 - (ttc / self.time_to_collision_threshold)
                risks.append({
                    'vehicle_id': vid,
                    'ttc': ttc,
                    'risk_level': risk_level,
                    'distance': distance,
                    'relative_speed': rel_speed
                })
        
        return risks
    
    def check_safe_overtaking(self, user_vehicle_id):
        """Check if it's safe to overtake."""
        if user_vehicle_id not in self.vehicle_history:
            return False, "User vehicle not detected"
        
        user_pos = np.array(self.vehicle_history[user_vehicle_id]['positions'][-1])
        user_metrics = self.calculate_vehicle_metrics(user_vehicle_id)
        
        if user_metrics is None:
            return False, "Unable to calculate user vehicle metrics"
        
        user_speed, user_direction, user_future = user_metrics
        
        # Check vehicles in adjacent lanes
        safe_to_overtake = True
        reason = "Safe to overtake"
        
        for vid, hist in self.vehicle_history.items():
            if vid == user_vehicle_id or len(hist['positions']) < 2:
                continue
            
            other_pos = np.array(hist['positions'][-1])
            other_metrics = self.calculate_vehicle_metrics(vid)
            
            if other_metrics is None:
                continue
                
            other_speed, other_direction, other_future = other_metrics
            
            # Check relative position and speed
            rel_pos = other_pos - user_pos
            distance = np.linalg.norm(rel_pos)
            
            # Check if vehicle is in overtaking zone
            if distance < self.safe_distance_front:
                if other_speed <= user_speed:
                    safe_to_overtake = False
                    reason = "Vehicle too close in front"
                    break
            
            # Check for vehicles in blind spots
            if abs(rel_pos[0]) < self.safe_distance_side and abs(rel_pos[1]) < self.safe_distance_front:
                safe_to_overtake = False
                reason = "Vehicle in blind spot"
                break
        
        return safe_to_overtake, reason
    
    def draw_visualization(self, frame, user_vehicle_id):
        """Draw visualization elements on the frame."""
        # Draw all vehicles
        for vid, hist in self.vehicle_history.items():
            if not hist['positions']:
                continue
                
            pos = hist['positions'][-1]
            bbox = hist['bbox']
            
            # Different colors for user vehicle and others
            color = (0, 255, 0) if vid == user_vehicle_id else (255, 0, 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                         (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                         color, 2)
            
            # Draw ID
            cv2.putText(frame, f"ID: {vid}", (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw predicted path
            metrics = self.calculate_vehicle_metrics(vid)
            if metrics is not None:
                _, _, future_pos = metrics
                cv2.line(frame, (int(pos[0]), int(pos[1])),
                        (int(future_pos[0]), int(future_pos[1])),
                        color, 1)
        
        # Draw collision risks
        risks = self.predict_collision_risk(user_vehicle_id)
        for risk in risks:
            other_pos = self.vehicle_history[risk['vehicle_id']]['positions'][-1]
            cv2.circle(frame, (int(other_pos[0]), int(other_pos[1])),
                      30, (0, 0, 255), 2)
            cv2.putText(frame, f"Risk: {risk['risk_level']:.2f}",
                       (int(other_pos[0]), int(other_pos[1]) - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw overtaking status
        safe, reason = self.check_safe_overtaking(user_vehicle_id)
        status_color = (0, 255, 0) if safe else (0, 0, 255)
        cv2.putText(frame, f"Overtaking: {reason}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   status_color, 2)
        
        return frame
    
    def process_frame(self, frame, user_vehicle_id=1):
        """Process a single frame and predict overtaking safety."""
        # Detect vehicles
        vehicles = self.detect_vehicles(frame)
        
        # Update tracking
        self.update_vehicle_tracking(vehicles)
        
        # Draw visualization
        output_frame = self.draw_visualization(frame.copy(), user_vehicle_id)
        
        # Get safety predictions
        safe_to_overtake, reason = self.check_safe_overtaking(user_vehicle_id)
        collision_risks = self.predict_collision_risk(user_vehicle_id)
        
        return output_frame, safe_to_overtake, reason, collision_risks
    
    def start_detection(self, video_source=0):
        """Start the overtaking prediction system."""
        cap = cv2.VideoCapture(video_source)
        user_vehicle_id = 1  # Assume first detected vehicle is user's vehicle
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            output_frame, safe, reason, risks = self.process_frame(frame, user_vehicle_id)
            
            # Display safety information
            if risks:
                print("\nCollision Risks Detected:")
                for risk in risks:
                    print(f"Vehicle {risk['vehicle_id']}: "
                          f"Time to Collision: {risk['ttc']:.2f}s, "
                          f"Risk Level: {risk['risk_level']:.2f}")
            
            # Show the frame
            cv2.imshow('Overtaking Prediction', output_frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    predictor = OvertakingPredictor()
    predictor.start_detection()

if __name__ == "__main__":
    main()
