import cv2
import numpy as np
import pytesseract
import time
import sys
import os

# Set the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class LicensePlateDetector:
    def __init__(self):
        # Check if Tesseract is installed
        try:
            pytesseract.get_tesseract_version()
            print("Tesseract OCR initialized successfully!")
        except Exception as e:
            print("Error: Tesseract OCR is not installed or not found in the specified path.")
            print("Please install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
            print("After installation, make sure to add it to your system PATH or update the path in the code.")
            sys.exit(1)

    def preprocess_image(self, img):
        """Preprocess the image to enhance text detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to remove noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Edge detection
        edges = cv2.Canny(filtered, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        return gray, edges, contours

    def find_plate_contour(self, contours):
        """Find the license plate contour."""
        plate_contour = None
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # License plate should be roughly rectangular (4 corners)
            if len(approx) == 4:
                plate_contour = approx
                break
        
        return plate_contour

    def extract_plate(self, img, plate_contour):
        """Extract the license plate region."""
        if plate_contour is None:
            return None, None
        
        # Get the minimum area rectangle
        rect = cv2.minAreaRect(plate_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Get width and height of the plate
        width = int(rect[1][0])
        height = int(rect[1][1])
        
        # Ensure width is greater than height
        if width < height:
            width, height = height, width
        
        # Get transformation points
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                           [0, 0],
                           [width-1, 0],
                           [width-1, height-1]], dtype="float32")
        
        # Apply perspective transform
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        plate = cv2.warpPerspective(img, matrix, (width, height))
        
        return plate, box

    def enhance_plate(self, plate):
        """Enhance the plate image for better text recognition."""
        if plate is None:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        
        # Resize image to a larger size for better OCR
        height, width = gray.shape
        gray = cv2.resize(gray, (width * 3, height * 3))
        
        # Apply bilateral filter for noise reduction while preserving edges
        denoised = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            19,
            9
        )
        
        # Remove small noise
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Connect broken characters
        kernel = np.ones((3,1), np.uint8)
        connected = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Invert back
        result = cv2.bitwise_not(connected)
        
        return result

    def read_plate_text(self, plate_img):
        """Read text from the plate image using Tesseract OCR."""
        if plate_img is None:
            return ""
        
        try:
            # Add white border around the image
            bordered = cv2.copyMakeBorder(plate_img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255,255,255])
            
            # Try different OCR configurations
            results = []
            
            # Configuration 1: Standard
            config1 = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text1 = pytesseract.image_to_string(bordered, config=config1).strip()
            if text1: results.append(text1)
            
            # Configuration 2: Single line
            config2 = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text2 = pytesseract.image_to_string(bordered, config=config2).strip()
            if text2: results.append(text2)
            
            # Configuration 3: Try with inverted image
            inverted = cv2.bitwise_not(bordered)
            text3 = pytesseract.image_to_string(inverted, config=config1).strip()
            if text3: results.append(text3)
            
            if not results:
                return ""
            
            # Get the result with the most alphanumeric characters
            text = max(results, key=lambda x: len([c for c in x if c.isalnum()]))
            
            # Clean up the text
            text = ''.join(c for c in text if c.isalnum())
            
            # Common OCR corrections for Indian license plates
            corrections = {
                '0': 'O',
                '1': 'I',
                '8': 'B',
                '5': 'S',
                '2': 'Z',
                '6': 'G',
                '4': 'A'
            }
            
            # Apply corrections
            cleaned_text = ""
            for i, char in enumerate(text):
                # First character is usually a letter
                if i == 0 and char in corrections and char.isdigit():
                    cleaned_text += corrections[char]
                # Numbers usually appear after the first two characters
                elif i <= 1 and char in corrections and char.isdigit():
                    cleaned_text += corrections[char]
                else:
                    cleaned_text += char
            
            return cleaned_text.upper()
            
        except Exception as e:
            print(f"Error reading text: {str(e)}")
            return ""

    def process_image(self, image_path):
        """Process a single image file."""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image file")
            
            # Process image
            gray, edges, contours = self.preprocess_image(img)
            plate_contour = self.find_plate_contour(contours)
            
            if plate_contour is not None:
                # Extract and enhance plate
                plate, box = self.extract_plate(img, plate_contour)
                enhanced_plate = self.enhance_plate(plate)
                
                # Read text
                text = self.read_plate_text(enhanced_plate)
                
                # Draw results on image
                cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
                cv2.putText(img, text, (box[0][0], box[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                return img, text
            
            return img, "No plate detected"
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None, str(e)

    def start_webcam(self):
        """Start real-time license plate detection."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not access webcam")
            return
            
        # Set lower resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Starting webcam detection. Press 'q' to quit.")
        print("Press 's' to save the current plate image for debugging")
        print("Press 'r' to reset the current detection")
        
        # Initialize variables for tracking detections
        detection_history = []
        max_history = 15  # Increased history size
        min_confidence = 3  # Increased minimum confidence
        save_counter = 0
        last_detection_time = time.time()
        
        # Variables to store stable detection
        stable_text = ""
        stable_confidence = 0
        frames_without_detection = 0
        max_frames_without_detection = 90  # Increased to 3 seconds at 30 fps
        
        # Create window
        cv2.namedWindow("License Plate Detection", cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Create a copy for display
                display_frame = frame.copy()
                
                # Process frame
                gray, edges, contours = self.preprocess_image(frame)
                plate_contour = self.find_plate_contour(contours)
                
                current_text = ""
                current_confidence = 0
                enhanced_plate = None
                
                if plate_contour is not None:
                    # Extract and enhance plate
                    plate, box = self.extract_plate(frame, plate_contour)
                    if plate is not None:
                        enhanced_plate = self.enhance_plate(plate)
                        
                        # Read text
                        text = self.read_plate_text(enhanced_plate)
                        
                        if text and len(text) >= 4:  # Only consider text with 4 or more characters
                            # Add to detection history
                            detection_history.append(text)
                            if len(detection_history) > max_history:
                                detection_history.pop(0)
                            
                            # Count occurrences of each detection
                            from collections import Counter
                            counts = Counter(detection_history)
                            most_common = counts.most_common(1)
                            
                            if most_common:
                                current_text = most_common[0][0]
                                current_confidence = most_common[0][1]
                                
                                # Update stable text if we have a better detection
                                if current_confidence >= min_confidence and (
                                    current_confidence > stable_confidence or
                                    (len(current_text) >= len(stable_text) and current_confidence >= stable_confidence)
                                ):
                                    if stable_text != current_text:  # Only update if different
                                        print(f"New stable detection: {current_text} (Confidence: {current_confidence})")
                                    stable_text = current_text
                                    stable_confidence = current_confidence
                                    frames_without_detection = 0
                                    last_detection_time = time.time()
                    
                    # Draw the plate contour
                    cv2.drawContours(display_frame, [plate_contour], -1, (0, 255, 0), 2)
                
                # Update frames counter
                if plate_contour is None or current_confidence < min_confidence:
                    frames_without_detection += 1
                
                # Only clear stable detection after a long period without detection
                if frames_without_detection > max_frames_without_detection:
                    if stable_text:
                        print("Lost detection")
                    stable_text = ""
                    stable_confidence = 0
                    detection_history.clear()
                
                # Create space for enhanced plate display
                h, w = display_frame.shape[:2]
                display_height = h + 200  # Add extra space for plate display
                combined_display = np.zeros((display_height, w, 3), dtype=np.uint8)
                combined_display[0:h, 0:w] = display_frame
                
                # Display the enhanced plate if available
                if enhanced_plate is not None:
                    # Convert enhanced_plate to color if it's grayscale
                    if len(enhanced_plate.shape) == 2:
                        enhanced_plate = cv2.cvtColor(enhanced_plate, cv2.COLOR_GRAY2BGR)
                    
                    # Resize enhanced plate to fit in the bottom section
                    plate_height = 180
                    aspect_ratio = enhanced_plate.shape[1] / enhanced_plate.shape[0]
                    plate_width = int(plate_height * aspect_ratio)
                    if plate_width > w:
                        plate_width = w
                        plate_height = int(w / aspect_ratio)
                    
                    resized_plate = cv2.resize(enhanced_plate, (plate_width, plate_height))
                    
                    # Calculate position to center the plate image
                    x_offset = (w - plate_width) // 2
                    y_offset = h + 10
                    
                    # Place the enhanced plate in the bottom section
                    combined_display[y_offset:y_offset+plate_height, x_offset:x_offset+plate_width] = resized_plate
                
                # Display the current state
                if stable_text:
                    # Calculate time since last detection
                    time_since_detection = time.time() - last_detection_time
                    confidence_percent = min(stable_confidence * 20, 100)
                    
                    # Show stable detection with confidence and time
                    cv2.putText(combined_display, f"Plate: {stable_text}",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.putText(combined_display, f"Confidence: {confidence_percent:.0f}%",
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # If we're currently detecting something different, show it too
                    if current_text and current_text != stable_text:
                        cv2.putText(combined_display, f"New: {current_text}",
                                  (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                elif current_text:
                    cv2.putText(combined_display, f"Reading: {current_text}",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Show instructions
                cv2.putText(combined_display, "Press 'q' to quit, 's' to save, 'r' to reset",
                          (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                          0.7, (0, 255, 0), 2)
                
                # Show processed frame
                cv2.imshow("License Plate Detection", combined_display)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and plate_contour is not None and enhanced_plate is not None:
                    # Save both original and enhanced images
                    filename_original = f"plate_original_{save_counter}.png"
                    filename_enhanced = f"plate_enhanced_{save_counter}.png"
                    cv2.imwrite(filename_original, plate)
                    cv2.imwrite(filename_enhanced, enhanced_plate)
                    print(f"Saved plate images as {filename_original} and {filename_enhanced}")
                    save_counter += 1
                elif key == ord('r'):
                    # Reset detection
                    stable_text = ""
                    stable_confidence = 0
                    detection_history.clear()
                    frames_without_detection = 0
                    print("Detection reset")
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    # Create detector instance
    detector = LicensePlateDetector()
    
    while True:
        print("\nLicense Plate Detection System")
        print("1. Process Image")
        print("2. Real-time Detection")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            # Process single image
            image_path = input("Enter image path: ")
            img, text = detector.process_image(image_path)
            if img is not None:
                print(f"\nDetected Text: {text}")
                cv2.imshow("Detected Plate", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        elif choice == '2':
            # Start real-time detection
            detector.start_webcam()
            
        elif choice == '3':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
