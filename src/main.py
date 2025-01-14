# src/main.py

import threading
import time
import signal
import sys
from core.camera import CameraManager
from core.vehicle import VehicleController
from detection.lane import LaneDetector
from viz.lane_viz import LaneVisualizer
from web.server import WebServer
from utils.types import DetectionFrame, ProcessingMetrics

class SelfDrivingSystem:
    """Main application that integrates all components of the self-driving system.
    
    This class orchestrates the interaction between the camera, vehicle control,
    lane detection, visualization, and web interface components. It manages the
    main processing loop and ensures smooth communication between all parts
    of the system.
    """
    
    def __init__(self):
        """Initialize all system components and set up communication channels."""
        print("Initializing Self-Driving System...")
        
        # Initialize core components
        self.camera = CameraManager()
        self.vehicle = VehicleController()
        self.lane_detector = LaneDetector()
        self.lane_visualizer = LaneVisualizer()
        
        # Initialize web interface with reference to this system
        self.web_server = WebServer(self)
        self.web_thread = None
        
        # System state
        self.running = False
        self.frame_count = 0
        self.latest_metrics = None
        
        # Thread synchronization
        self.metrics_lock = threading.Lock()
        
        # Set up signal handling for clean shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
    def start(self):
        """Start all system components and begin main processing loop."""
        print("Starting Self-Driving System...")
        
        # Start web server in separate thread
        self.web_thread = threading.Thread(target=self.web_server.run)
        self.web_thread.daemon = True
        self.web_thread.start()
        print("Web interface started at http://localhost:5000")
        
        # Start camera
        self.camera.start()
        print("Camera system initialized")
        
        # Main control loop
        self.running = True
        self._main_loop()
        
    def _main_loop(self):
        """Main processing loop that handles frame processing and vehicle control."""
        while self.running:
            # Get frame from camera
            frame = self.camera.get_frame()
            if frame is None:
                continue
            
            # Create detection frame
            self.frame_count += 1
            detection_frame = DetectionFrame(
                frame=frame,
                timestamp=time.time(),
                frame_id=self.frame_count,
                width=frame.shape[1],
                height=frame.shape[0]
            )
            
            # Process frame for lane detection
            detection_result = self.lane_detector.process_frame(detection_frame)
            
            # Create visualization
            viz_result = self.lane_visualizer.create_visualization(
                frame=frame,
                detection_result=detection_result,
                debug=self.web_server.debug_enabled
            )
            
            # Update web interface
            self.web_server.update_visualization(viz_result)
            
            # Update metrics
            with self.metrics_lock:
                self.latest_metrics = viz_result.metrics
            
            # Update vehicle control if autonomous mode is enabled
            if self.web_server.autonomous_enabled and detection_result.is_valid:
                self._update_vehicle_control(detection_result)
            else:
                self.vehicle.stop()
            
            # Small delay to prevent CPU overload
            time.sleep(0.01)
    
    def _update_vehicle_control(self, detection_result):
        """Update vehicle control based on lane detection results.
        
        This method implements curve-aware control, adjusting both steering
        and speed based on the detected lane curvature.
        
        Args:
            detection_result: Current lane detection results
        """
        lane_data = detection_result.data
        
        # Update vehicle control with curve handling
        self.vehicle.update_curve_control(
            curve_radius=lane_data.curve_radius,
            curve_direction=lane_data.curve_direction,
            center_offset=lane_data.center_offset
        )
    
    def get_latest_metrics(self) -> ProcessingMetrics:
        """Thread-safe access to latest processing metrics."""
        with self.metrics_lock:
            return self.latest_metrics
    
    def _handle_shutdown(self, signum, frame):
        """Handle system shutdown gracefully.
        
        Ensures all components are properly stopped and resources are released.
        """
        print("\nInitiating system shutdown...")
        self.running = False
        self.vehicle.stop()
        self.camera.stop()
        print("System shutdown complete.")
        sys.exit(0)

if __name__ == "__main__":
    # Create and start the self-driving system
    system = SelfDrivingSystem()
    system.start()