import cv2
import numpy as np
import tensorflow as tf
from skimage import transform, exposure
from ultralytics import YOLO
from time import time
from PIL import Image
from customtkinter import CTkImage

# Import utilities
from utils.class_names import getClassName
from utils.image_processing import image_processing

class TrafficSignRecognizer:
    def __init__(self, yolo_model_path, cnn_model_path):
        """
        Initialize the traffic sign recognition system
        
        Args:
            yolo_model_path: Path to YOLO model (.pt file)
            cnn_model_path: Path to CNN model (SavedModel directory)
        """
        print("🚀 Initializing Traffic Sign Recognition System...")
        
        # Load YOLO model for detection
        t1 = time()
        self.yolo_model = YOLO(yolo_model_path)
        print(f"✅ YOLO model loaded in {time() - t1:.2f}s")
        
        # Load CNN model for classification
        t1 = time()
        self.cnn_model = tf.saved_model.load(cnn_model_path)
        print(f"✅ CNN model loaded in {time() - t1:.2f}s")
        
        self.confidence_threshold = 0.5
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        print("🎯 System ready for inference!")
    
    def predict_with_cnn(self, image):
        """
        Classify traffic sign using CNN model
        """
        # Preprocess image
        processed_image = image_processing(image)
        processed_image = processed_image.reshape(1, 32, 32, 3).astype(np.float32)
        
        # Convert to tensor and predict
        input_tensor = tf.constant(processed_image)
        outputs = self.cnn_model(input_tensor)
        
        # Extract predictions
        predictions = outputs.numpy()
        class_index = np.argmax(predictions[0])
        probability = np.max(predictions[0])
        
        return class_index, probability
    
    def process_frame(self, frame):
        """
        Process a single frame for traffic sign recognition
        
        Returns:
            processed_frame: Frame with detections drawn
            detections: List of detection information
        """
        detections = []
        processed_frame = frame.copy()
        
        # Step 1: YOLO Detection
        results = self.yolo_model(frame, verbose=False, conf=0.5)
        
        if len(results) > 0 and results[0].boxes is not None:
            # Get YOLO visualization
            processed_frame = results[0].plot()
            boxes = results[0].boxes
            
            # Process each detection
            for box in boxes:
                try:
                    # Extract bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    yolo_confidence = box.conf[0].item()
                    
                    # Validate bounding box
                    if (x2 > x1 and y2 > y1 and 
                        x1 >= 0 and y1 >= 0 and 
                        x2 <= frame.shape[1] and y2 <= frame.shape[0]):
                        
                        # Extract Region of Interest (ROI)
                        roi = frame[y1:y2, x1:x2]
                        
                        if roi.size > 0 and roi.shape[0] >= 5 and roi.shape[1] >= 5:
                            # Step 2: CNN Classification
                            class_index, probability = self.predict_with_cnn(roi)
                            
                            if probability > self.confidence_threshold:
                                class_name = getClassName(class_index)
                                
                                # Store detection info
                                detections.append({
                                    'bbox': (x1, y1, x2, y2),
                                    'class_id': class_index,
                                    'class_name': class_name,
                                    'confidence': probability,
                                    'yolo_confidence': yolo_confidence
                                })
                                
                                # Enhanced visualization
                                self._draw_detection(processed_frame, x1, y1, x2, y2, 
                                                   class_name, probability)
                                
                                print(f"🚦 Detected: {class_name} ({probability*100:.1f}%)")
                                
                except Exception as e:
                    print(f"⚠️ Detection error: {e}")
        
        return processed_frame, detections
    
    def _draw_detection(self, frame, x1, y1, x2, y2, class_name, probability):
        """Draw detection information on frame"""
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw info background
        info_bg_y = max(y1 - 60, 10)
        cv2.rectangle(frame, (x1, info_bg_y), (x2, y1), (0, 255, 0), -1)
        cv2.rectangle(frame, (x1, info_bg_y), (x2, y1), (0, 255, 0), 2)
        
        # Draw class name
        cv2.putText(frame, class_name, 
                   (x1 + 5, y1 - 40), self.font, 0.5, (0, 0, 0), 2)
        
        # Draw probability
        cv2.putText(frame, f'Confidence: {probability*100:.1f}%', 
                   (x1 + 5, y1 - 15), self.font, 0.5, (0, 0, 0), 2)

def start_webcam_inference(condition_prerecorded=False, camgui=None):
    """
    Main inference function for GUI
    
    Args:
        condition_prerecorded: Use webcam (False) or video file (True)
        camgui: GUI object for displaying results
    """
    # Initialize recognizer with your model paths
    recognizer = TrafficSignRecognizer(
        yolo_model_path="./models/yolo/best.pt",
        cnn_model_path="./models/cnn/saved_model"
    )
    
    # Setup video capture
    if condition_prerecorded:
        cap = cv2.VideoCapture("video.mp4")
        print("📹 Processing video file...")
    else:
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)  # Width
        cap.set(4, 640)  # Height
        print("📷 Starting webcam inference...")
    
    print("\n🎮 Controls:")
    print("   - Press 'q' to quit")
    print("   - Press 'p' to pause/resume")
    
    paused = False
    frame_count = 0
    fps_time = time()
    
    try:
        while True:
            if not paused:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ End of video or capture error")
                    break
                
                frame_count += 1
                
                # Process frame
                processed_frame, detections = recognizer.process_frame(frame)
                
                # Calculate FPS
                if frame_count % 30 == 0:
                    fps = 30 / (time() - fps_time)
                    cv2.putText(processed_frame, f'FPS: {fps:.1f}', (10, 30), 
                              recognizer.font, 0.7, (255, 255, 255), 2)
                    fps_time = time()
                
                # Display results
                if camgui is None:
                    # Show in OpenCV window
                    cv2.imshow('Traffic Sign Recognition', processed_frame)
                else:
                    # Send to GUI
                    self._update_gui(camgui, processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("⏸️ Paused" if paused else "▶️ Resumed")
                
    except Exception as e:
        print(f"❌ Inference error: {e}")
    finally:
        # Cleanup
        cap.release()
        if camgui is None:
            cv2.destroyAllWindows()
        print("✅ Inference session ended.")
    
    def _update_gui(self, camgui, processed_frame):
        """Update GUI with processed frame"""
        try:
            # Convert BGR to RGB for PIL
            cvimage = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGBA)
            image = Image.fromarray(cvimage)
            imgctk = CTkImage(image, size=(640, 480))
            camgui.image_label.configure(image=imgctk)
            camgui.update()
        except Exception as e:
            print(f"⚠️ GUI update error: {e}")
