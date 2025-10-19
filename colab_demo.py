"""
Google Colab compatible version of Traffic Sign Recognition
"""
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import os
from google.colab.patches import cv2_imshow
from utils.class_names import getClassName
from utils.image_processing import image_processing

class ColabTrafficSignRecognizer:
    def __init__(self):
        """Initialize for Colab environment"""
        print("🚀 Initializing Traffic Sign Recognition for Colab...")
        
        # Load models (update paths for Colab)
        self.yolo_model = YOLO('/content/models/yolo/best.pt')
        self.cnn_model = tf.saved_model.load('/content/models/cnn/saved_model')
        
        print("✅ Models loaded successfully!")
    
    def process_image(self, image_path):
        """Process a single image and display results"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return
        
        print(f"📷 Processing: {image_path}")
        
        # YOLO detection
        results = self.yolo_model(image, verbose=False)
        processed_image = image.copy()
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0].item()
                    
                    # Extract ROI
                    roi = image[y1:y2, x1:x2]
                    
                    if roi.size > 0:
                        # CNN classification
                        class_id, prob = self._classify_with_cnn(roi)
                        
                        if prob > 0.5:
                            class_name = getClassName(class_id)
                            detections.append({
                                'class': class_name,
                                'confidence': f"{prob*100:.1f}%",
                                'bbox': (x1, y1, x2, y2)
                            })
                            
                            # Draw on image
                            cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(processed_image, f"{class_name} {prob*100:.1f}%", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                except Exception as e:
                    print(f"⚠️ Error in detection: {e}")
        
        # Display results
        print(f"\n🎯 Detections: {len(detections)}")
        for det in detections:
            print(f"   - {det['class']} ({det['confidence']})")
        
        # Show original and processed images
        print("\n🖼️ Original Image:")
        cv2_imshow(image)
        
        print("\n🔍 Processed Image with Detections:")
        cv2_imshow(processed_image)
        
        # Save result
        output_path = "/content/result.jpg"
        cv2.imwrite(output_path, processed_image)
        print(f"\n💾 Result saved to: {output_path}")
        
        return detections
    
    def process_video(self, video_path):
        """Process video and save output"""
        print(f"🎥 Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Could not open video")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        output_path = "/content/output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        print("⏳ Processing video frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = self.yolo_model(frame, verbose=False)
            processed_frame = frame.copy()
            
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        roi = frame[y1:y2, x1:x2]
                        
                        if roi.size > 0:
                            class_id, prob = self._classify_with_cnn(roi)
                            if prob > 0.5:
                                class_name = getClassName(class_id)
                                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(processed_frame, f"{class_name} {prob*100:.1f}%", 
                                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    except:
                        pass
            
            out.write(processed_frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"📊 Processed {frame_count} frames...")
        
        cap.release()
        out.release()
        print(f"✅ Video processing complete! Saved to: {output_path}")
    
    def _classify_with_cnn(self, image):
        """Classify image using CNN model"""
        processed_image = image_processing(image)
        processed_image = processed_image.reshape(1, 32, 32, 3).astype(np.float32)
        
        input_tensor = tf.constant(processed_image)
        outputs = self.cnn_model(input_tensor)
        predictions = outputs.numpy()
        
        class_index = np.argmax(predictions[0])
        probability = np.max(predictions[0])
        
        return class_index, probability

def demo():
    """Run a demo with sample images"""
    recognizer = ColabTrafficSignRecognizer()
    
    print("\n" + "="*50)
    print("🚦 TRAFFIC SIGN RECOGNITION - COLAB DEMO")
    print("="*50)
    
    # Check for sample images
    sample_images = []
    for img in os.listdir('/content'):
        if img.lower().endswith(('.jpg', '.jpeg', '.png')):
            sample_images.append(f'/content/{img}')
    
    if sample_images:
        print(f"📁 Found {len(sample_images)} images to process")
        for img_path in sample_images:
            recognizer.process_image(img_path)
            print("\n" + "-"*40 + "\n")
    else:
        print("📸 No images found. Please upload some traffic sign images!")
        print("You can upload images using:")
        print("""
        from google.colab import files
        uploaded = files.upload()
        """)

if __name__ == "__main__":
    demo()
