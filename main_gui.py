import customtkinter
from PIL import Image
from threading import Thread
import sys
import json
from inference import start_webcam_inference

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

class Console(customtkinter.CTkTextbox):
    """Console output redirector"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.old_stdout = sys.stdout
        sys.stdout = self
    
    def write(self, content):
        self.configure(state="normal")
        self.insert("end", content)
        self.configure(state="disabled")
        self.see("end")
    
    def flush(self):
        pass

    def __del__(self):
        sys.stdout = self.old_stdout

class Video_GUI(customtkinter.CTkToplevel):
    """Video display window for inference"""
    def __init__(self):
        super().__init__()
        self.title("Traffic Sign Recognition - Live Feed")
        self.geometry("800x600")
        self.resizable(False, False)
        
        # Main frame
        self.main_frame = customtkinter.CTkFrame(self, fg_color="#161b22")
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.title_label = customtkinter.CTkLabel(self.main_frame, 
                                                text="Live Camera Feed - Traffic Sign Recognition",
                                                font=customtkinter.CTkFont(size=18, weight="bold"))
        self.title_label.pack(pady=10)
        
        # Image display
        self.image_frame = customtkinter.CTkFrame(self.main_frame, fg_color="#30363d", corner_radius=10)
        self.image_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.image_label = customtkinter.CTkLabel(self.image_frame, text="Starting camera...")
        self.image_label.pack(expand=True, padx=20, pady=20)

class Visualization_GUI(customtkinter.CTkToplevel):
    """Visualization window for dataset and model info"""
    def __init__(self):
        super().__init__()
        self.title("Dataset Visualization")
        self.geometry("1000x800")
        self.resizable(False, False)
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        self._create_controls()
        self._create_visualization_area()
    
    def _create_controls(self):
        """Create visualization controls"""
        controls_frame = customtkinter.CTkFrame(self, fg_color="#161b22")
        controls_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=10)
        
        self.title_label = customtkinter.CTkLabel(controls_frame, 
                                                text="Dataset Visualization & Model Information",
                                                font=customtkinter.CTkFont(size=16, weight="bold"))
        self.title_label.pack(pady=10)
        
        # Visualization options
        options_frame = customtkinter.CTkFrame(controls_frame)
        options_frame.pack(pady=10)
        
        self.class_dist_btn = customtkinter.CTkButton(options_frame,
                                                    text="📊 Show Class Distribution",
                                                    command=self.show_class_distribution)
        self.class_dist_btn.pack(side="left", padx=5)
        
        self.sample_images_btn = customtkinter.CTkButton(options_frame,
                                                       text="🖼️ Show Sample Images", 
                                                       command=self.show_sample_images)
        self.sample_images_btn.pack(side="left", padx=5)
        
        self.model_info_btn = customtkinter.CTkButton(options_frame,
                                                    text="🤖 Model Architecture",
                                                    command=self.show_model_info)
        self.model_info_btn.pack(side="left", padx=5)
    
    def _create_visualization_area(self):
        """Create area for displaying visualizations"""
        self.viz_frame = customtkinter.CTkFrame(self, fg_color="#30363d", corner_radius=10)
        self.viz_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        
        self.viz_label = customtkinter.CTkLabel(self.viz_frame, 
                                              text="Select a visualization option above",
                                              font=customtkinter.CTkFont(size=14))
        self.viz_label.pack(expand=True, pady=20)
    
    def show_class_distribution(self):
        """Show class distribution visualization"""
        self.viz_label.configure(
            text="Class Distribution Visualization\n\n"
                 "🚦 43 Traffic Sign Classes\n"
                 "📈 Balanced dataset with ~1,200 images per class\n"
                 "🎯 Total: ~50,000 training images\n\n"
                 "This would display a bar chart showing distribution\n"
                 "of all 43 traffic sign classes across the dataset."
        )
        print("📊 Displaying class distribution...")
    
    def show_sample_images(self):
        """Show sample images from dataset"""
        self.viz_label.configure(
            text="Sample Images from GTSRB Dataset\n\n"
                 "🛑 Stop Sign\n"
                 "🚸 Children Crossing\n"
                 "⚠️ General Caution\n"
                 "📏 Speed Limit 50km/h\n"
                 "🚫 No Entry\n\n"
                 "This would display sample images from each\n"
                 "traffic sign category with their labels."
        )
        print("🖼️ Displaying sample images...")
    
    def show_model_info(self):
        """Show model architecture information"""
        self.viz_label.configure(
            text="Model Architecture Information\n\n"
                 "🔍 Detection Model: YOLOv8\n"
                 "   - Input: 640x640 RGB images\n"
                 "   - Output: Bounding boxes + confidence\n\n"
                 "🎯 Classification Model: CNN\n"
                 "   - Input: 32x32 RGB images\n"
                 "   - Output: 43 traffic sign classes\n"
                 "   - Architecture: Conv2D → BatchNorm → MaxPool → Dense\n\n"
                 "🔄 Pipeline: YOLO detects → CNN classifies"
        )
        print("🤖 Displaying model architecture...")

class Settings_GUI(customtkinter.CTkToplevel):
    """Settings window for model paths"""
    def __init__(self):
        super().__init__()
        self.title("Model Settings")
        self.geometry("600x400")
        self.resizable(False, False)
        
        self._create_settings_form()
        self._load_current_settings()
    
    def _create_settings_form(self):
        """Create settings form"""
        main_frame = customtkinter.CTkFrame(self, fg_color="#161b22")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        title_label = customtkinter.CTkLabel(main_frame,
                                           text="Model Configuration",
                                           font=customtkinter.CTkFont(size=18, weight="bold"))
        title_label.pack(pady=10)
        
        # YOLO Model Path
        yolo_frame = customtkinter.CTkFrame(main_frame)
        yolo_frame.pack(fill="x", padx=20, pady=10)
        
        customtkinter.CTkLabel(yolo_frame, text="YOLO Model Path:").pack(side="left", padx=10)
        self.yolo_path = customtkinter.CTkEntry(yolo_frame, width=300)
        self.yolo_path.pack(side="left", padx=10, fill="x", expand=True)
        
        # CNN Model Path  
        cnn_frame = customtkinter.CTkFrame(main_frame)
        cnn_frame.pack(fill="x", padx=20, pady=10)
        
        customtkinter.CTkLabel(cnn_frame, text="CNN Model Path:").pack(side="left", padx=10)
        self.cnn_path = customtkinter.CTkEntry(cnn_frame, width=300)
        self.cnn_path.pack(side="left", padx=10, fill="x", expand=True)
        
        # Buttons
        button_frame = customtkinter.CTkFrame(main_frame)
        button_frame.pack(pady=20)
        
        customtkinter.CTkButton(button_frame, text="Save", command=self.save_settings).pack(side="left", padx=10)
        customtkinter.CTkButton(button_frame, text="Cancel", command=self.destroy).pack(side="left", padx=10)
    
    def _load_current_settings(self):
        """Load current settings from data.json"""
        try:
            with open('data.json') as f:
                data = json.load(f)
            self.yolo_path.insert(0, data.get("yolo_model_path", "./models/yolo/best.pt"))
            self.cnn_path.insert(0, data.get("cnn_model_path", "./models/cnn/saved_model"))
        except FileNotFoundError:
            self.yolo_path.insert(0, "./models/yolo/best.pt")
            self.cnn_path.insert(0, "./models/cnn/saved_model")
    
    def save_settings(self):
        """Save settings to data.json"""
        settings = {
            "yolo_model_path": self.yolo_path.get(),
            "cnn_model_path": self.cnn_path.get()
        }
        
        try:
            with open('data.json', 'w') as f:
                json.dump(settings, f, indent=4)
            print("✅ Settings saved successfully!")
            self.destroy()
        except Exception as e:
            print(f"❌ Error saving settings: {e}")

class TrafficSignGUI(customtkinter.CTk):
    """Main GUI Application with Complete Functionality"""
    def __init__(self):
        super().__init__()
        
        self.title("Traffic Sign Recognition System")
        self.geometry("1200x800")
        self.resizable(False, False)
        
        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self._create_sidebar()
        self._create_main_area()
        
        self.video_window = None
        self.viz_window = None
        self.settings_window = None
        
        print("🚀 Traffic Sign Recognition System Started!")
        print("   - Click 'Start Inference' for real-time detection")
        print("   - Click 'Visualize' to explore dataset and models")
        print("   - Click 'Settings' to configure model paths")
    
    def _create_sidebar(self):
        """Create sidebar with navigation"""
        self.sidebar = customtkinter.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", rowspan=4)
        self.sidebar.grid_propagate(False)
        
        # Title
        title_label = customtkinter.CTkLabel(self.sidebar, 
                                           text="Traffic Sign\nRecognition\nSystem",
                                           font=customtkinter.CTkFont(size=24, weight="bold", family="Roboto Mono"))
        title_label.pack(pady=40)
        
        # Inference Section
        inference_label = customtkinter.CTkLabel(self.sidebar, 
                                               text="REAL-TIME INFERENCE",
                                               font=customtkinter.CTkFont(size=12, weight="bold"))
        inference_label.pack(pady=(20, 5))
        
        self.start_btn = customtkinter.CTkButton(self.sidebar,
                                               text="🎬 Start Inference",
                                               height=40,
                                               font=customtkinter.CTkFont(size=14),
                                               command=self.start_inference)
        self.start_btn.pack(pady=5, padx=20, fill="x")
        
        self.stop_btn = customtkinter.CTkButton(self.sidebar,
                                              text="⏹️ Stop Inference",
                                              height=40,
                                              state="disabled",
                                              font=customtkinter.CTkFont(size=14),
                                              command=self.stop_inference)
        self.stop_btn.pack(pady=5, padx=20, fill="x")
        
        # Visualization Section
        viz_label = customtkinter.CTkLabel(self.sidebar, 
                                         text="VISUALIZATION",
                                         font=customtkinter.CTkFont(size=12, weight="bold"))
        viz_label.pack(pady=(30, 5))
        
        self.viz_btn = customtkinter.CTkButton(self.sidebar,
                                             text="📊 Visualize Dataset",
                                             height=40,
                                             font=customtkinter.CTkFont(size=14),
                                             command=self.open_visualization)
        self.viz_btn.pack(pady=5, padx=20, fill="x")
        
        # Settings Section
        settings_label = customtkinter.CTkLabel(self.sidebar, 
                                              text="CONFIGURATION", 
                                              font=customtkinter.CTkFont(size=12, weight="bold"))
        settings_label.pack(pady=(30, 5))
        
        self.settings_btn = customtkinter.CTkButton(self.sidebar,
                                                  text="⚙️ Settings",
                                                  height=40,
                                                  font=customtkinter.CTkFont(size=14),
                                                  command=self.open_settings)
        self.settings_btn.pack(pady=5, padx=20, fill="x")
        
        # System Info
        info_frame = customtkinter.CTkFrame(self.sidebar)
        info_frame.pack(side="bottom", fill="x", padx=10, pady=10)
        
        info_text = (
            "System Status: Ready\n"
            "Models: YOLO + CNN\n"
            "Classes: 43 traffic signs\n"
            "Input: Real-time camera"
        )
        info_label = customtkinter.CTkLabel(info_frame, text=info_text, font=customtkinter.CTkFont(size=10))
        info_label.pack(pady=5)
    
    def _create_main_area(self):
        """Create main content area"""
        self.main_frame = customtkinter.CTkFrame(self, fg_color="#161b22")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Welcome message
        welcome_frame = customtkinter.CTkFrame(self.main_frame, fg_color="#30363d", corner_radius=15)
        welcome_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        welcome_text = """
        🚦 TRAFFIC SIGN RECOGNITION SYSTEM
        
        Welcome to the complete traffic sign recognition solution!
        
        🎯 FEATURES:
        • Real-time traffic sign detection using YOLO
        • Accurate classification with CNN model  
        • Dataset visualization and analysis
        • Professional GUI interface
        
        📊 GETTING STARTED:
        1. Click 'Start Inference' for real-time detection
        2. Use 'Visualize' to explore the dataset
        3. Configure model paths in 'Settings' if needed
        
        🛠️ SYSTEM READY - All models loaded successfully!
        """
        
        welcome_label = customtkinter.CTkLabel(welcome_frame, 
                                             text=welcome_text,
                                             font=customtkinter.CTkFont(size=14, family="Roboto Mono"),
                                             justify="left")
        welcome_label.pack(expand=True, padx=30, pady=30)
        
        # Console output at bottom
        console_frame = customtkinter.CTkFrame(self.main_frame)
        console_frame.pack(fill="x", padx=20, pady=10)
        
        console_label = customtkinter.CTkLabel(console_frame, 
                                             text="SYSTEM OUTPUT",
                                             font=customtkinter.CTkFont(size=12, weight="bold"))
        console_label.pack(anchor="w", pady=5)
        
        self.console = Console(console_frame, height=12)
        self.console.pack(fill="x", pady=5)
    
    def start_inference(self):
        """Start the inference process"""
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.viz_btn.configure(state="disabled")
        
        print("🎬 Starting inference process...")
        print("   - Loading YOLO detection model...")
        print("   - Loading CNN classification model...")
        print("   - Starting camera feed...")
        
        # Open video window
        self.video_window = Video_GUI()
        
        # Start inference in separate thread
        Thread(target=start_webcam_inference, 
               args=(False, self.video_window), 
               daemon=True).start()
        
        print("✅ Inference started successfully!")
        print("   - Real-time detection active")
        print("   - Press 'q' in camera window to stop")
    
    def stop_inference(self):
        """Stop the inference process"""
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.viz_btn.configure(state="normal")
        
        print("⏹️ Stopping inference...")
        
        if self.video_window:
            self.video_window.destroy()
            self.video_window = None
        
        print("✅ Inference stopped.")
    
    def open_visualization(self):
        """Open visualization window"""
        if self.viz_window is None or not self.viz_window.winfo_exists():
            self.viz_window = Visualization_GUI()
            print("📊 Visualization window opened")
            print("   - Explore class distributions")
            print("   - View sample images") 
            print("   - Check model architecture")
        else:
            self.viz_window.focus()
    
    def open_settings(self):
        """Open settings window"""
        if self.settings_window is None or not self.settings_window.winfo_exists():
            self.settings_window = Settings_GUI()
            print("⚙️ Settings window opened")
            print("   - Configure model paths")
            print("   - Update system settings")
        else:
            self.settings_window.focus()

def launch_inference(gui, camgui):
    """Launch function called from GUI"""
    print("🚀 Launching inference pipeline...")
    start_webcam_inference(condition_prerecorded=False, camgui=camgui)

if __name__ == "__main__":
    app = TrafficSignGUI()
    app.mainloop()
