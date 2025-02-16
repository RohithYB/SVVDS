import cv2
import numpy as np
from collections import deque
import time

class LaneChangeDetector:
    def __init__(self):
        # Initialize parameters for lane detection
        self.lane_history = {}  # Track vehicle positions relative to lanes
        self.vehicle_positions = {}  # Store vehicle centroids
        self.lane_change_cooldown = {}  # Prevent multiple detections
        self.cooldown_time = 3  # Seconds between lane change detections
        
        # Parameters for lane detection
        self.lane_detection_params = {
            'canny_low': 50,
            'canny_high': 150,
            'rho': 1,
            'theta': np.pi/180,
            'threshold': 50,
            'min_line_length': 50,
            'max_line_gap': 30
        }
        
        # Region of interest vertices (will be set based on frame size)
        self.roi_vertices = None
        
    def preprocess_frame(self, frame):
        """Preprocess the frame for lane detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges using Canny
        edges = cv2.Canny(blurred, 
                         self.lane_detection_params['canny_low'],
                         self.lane_detection_params['canny_high'])
        
        # Apply region of interest mask
        if self.roi_vertices is None:
            self.set_roi(frame.shape)
        
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [self.roi_vertices], 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        return masked_edges
    
    def set_roi(self, frame_shape):
        """Set region of interest vertices based on frame size."""
        height, width = frame_shape[:2]
        self.roi_vertices = np.array([
            [(0, height),
             (width * 0.45, height * 0.6),
             (width * 0.55, height * 0.6),
             (width, height)]
        ], dtype=np.int32)
    
    def detect_lanes(self, edges):
        """Detect lane lines using Hough transform."""
        lines = cv2.HoughLinesP(
            edges,
            self.lane_detection_params['rho'],
            self.lane_detection_params['theta'],
            self.lane_detection_params['threshold'],
            minLineLength=self.lane_detection_params['min_line_length'],
            maxLineGap=self.lane_detection_params['max_line_gap']
        )
        
        if lines is None:
            return []
        
        # Separate lines into left and right lanes based on slope
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:  # Avoid division by zero
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            
            if abs(slope) < 0.5:  # Filter out horizontal lines
                continue
                
            if slope < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)
        
        return left_lines, right_lines
    
    def detect_vehicles(self, frame):
        """Detect vehicles using background subtraction and contour detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction (simplified version)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area
        min_area = 1000  # Minimum contour area to be considered a vehicle
        vehicles = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                vehicles.append({
                    'bbox': (x, y, w, h),
                    'centroid': (x + w//2, y + h//2)
                })
        
        return vehicles
    
    def detect_lane_changes(self, frame, vehicles):
        """Detect lane changes for each vehicle."""
        current_time = time.time()
        lane_changes = []
        
        # Process each vehicle
        for idx, vehicle in enumerate(vehicles):
            centroid = vehicle['centroid']
            
            # Initialize vehicle tracking if new
            if idx not in self.vehicle_positions:
                self.vehicle_positions[idx] = deque(maxlen=10)
                self.lane_history[idx] = deque(maxlen=2)
                self.lane_change_cooldown[idx] = 0
            
            # Update vehicle position history
            self.vehicle_positions[idx].append(centroid)
            
            # Determine current lane (simplified: left, middle, right)
            frame_width = frame.shape[1]
            current_lane = None
            x = centroid[0]
            
            if x < frame_width * 0.33:
                current_lane = 'left'
            elif x < frame_width * 0.66:
                current_lane = 'middle'
            else:
                current_lane = 'right'
            
            # Check for lane change
            if len(self.lane_history[idx]) > 0:
                previous_lane = self.lane_history[idx][-1]
                if (current_lane != previous_lane and 
                    current_time - self.lane_change_cooldown[idx] > self.cooldown_time):
                    lane_changes.append({
                        'vehicle_id': idx,
                        'from_lane': previous_lane,
                        'to_lane': current_lane,
                        'position': centroid
                    })
                    self.lane_change_cooldown[idx] = current_time
            
            self.lane_history[idx].append(current_lane)
        
        return lane_changes
    
    def draw_visualization(self, frame, vehicles, lane_changes, lanes):
        """Draw visualization elements on the frame."""
        # Draw lanes
        if lanes:
            left_lines, right_lines = lanes
            for line in left_lines + right_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw vehicles
        for vehicle in vehicles:
            x, y, w, h = vehicle['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Draw lane change alerts
        for change in lane_changes:
            pos = change['position']
            cv2.circle(frame, pos, 10, (0, 0, 255), -1)
            text = f"Lane Change: {change['from_lane']} -> {change['to_lane']}"
            cv2.putText(frame, text, (pos[0] - 100, pos[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
    def process_frame(self, frame):
        """Process a single frame and detect lane changes."""
        # Preprocess frame for lane detection
        edges = self.preprocess_frame(frame)
        
        # Detect lanes
        lanes = self.detect_lanes(edges)
        
        # Detect vehicles
        vehicles = self.detect_vehicles(frame)
        
        # Detect lane changes
        lane_changes = self.detect_lane_changes(frame, vehicles)
        
        # Draw visualization
        output_frame = self.draw_visualization(frame.copy(), vehicles, lane_changes, lanes)
        
        return output_frame, lane_changes
    
    def start_detection(self, video_source=0):
        """Start the lane change detection process."""
        cap = cv2.VideoCapture(video_source)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            output_frame, lane_changes = self.process_frame(frame)
            
            # Display lane change events
            for change in lane_changes:
                print(f"Vehicle {change['vehicle_id']} changed from "
                      f"{change['from_lane']} to {change['to_lane']} lane")
            
            # Show the frame
            cv2.imshow('Lane Change Detection', output_frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = LaneChangeDetector()
    detector.start_detection()

if __name__ == "__main__":
    main()
