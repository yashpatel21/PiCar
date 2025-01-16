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
        self.autonomous_enabled = False  # Track autonomous state at system level
        
        # Thread synchronization
        self.metrics_lock = threading.Lock()
        self.autonomous_lock = threading.Lock()  # Add lock for autonomous state
        
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
        self.autonomous_enabled = False
        self._main_loop()
    
    def set_autonomous_mode(self, enabled: bool) -> None:
        """Thread-safe method to toggle autonomous mode.
        
        This method ensures clean transitions between manual and autonomous modes
        by properly handling vehicle control state.
        """
        with self.autonomous_lock:
            self.autonomous_enabled = enabled
            if not enabled:
                # Safely stop vehicle when disabling autonomous mode
                self.vehicle.stop()
        
    def _main_loop(self):
        """Main processing loop that handles frame processing and vehicle control.
        
        The loop runs continuously for video processing and visualization, but
        only controls the vehicle when autonomous mode is enabled through the
        web interface.
        """
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
            
            # Check autonomous mode state in thread-safe way
            with self.autonomous_lock:
                autonomous_active = self.autonomous_enabled
            
            # Update vehicle control if autonomous mode is enabled
            if autonomous_active:
                if detection_result.is_valid:
                    # Normal lane following when we have valid detection
                    self._update_vehicle_control(detection_result)
                else:
                    # If we lose lane detection, gradually slow down for safety
                    # but maintain previous steering direction briefly
                    current_speed = self.vehicle.state.speed
                    if current_speed > self.vehicle.min_curve_speed:
                        self.vehicle.set_speed(current_speed * 0.8)  # Gradual slowdown
                    else:
                        self.vehicle.stop()
            else:
                # Ensure vehicle is stopped when not in autonomous mode
                self.vehicle.stop()
            
            # Small delay to prevent CPU overload
            time.sleep(0.01)

    def _update_vehicle_control(self, detection_result):
        """Update vehicle control based on lane detection results.
        
        This method implements our lane-following strategy by:
        1. Getting steering angles from the lane detector based on detected curves
        2. Passing those angles along with curve data to the vehicle controller
        3. Letting the vehicle controller handle the physics of speed and steering
        
        The lane detector has already considered various scenarios like:
        - Both lanes visible (centering between lanes)
        - Single lane visible (maintaining offset and handling curves)
        - Sharp turns (exponential steering response)
        
        Args:
            detection_result: The results from lane detection including curves,
                            detection status, and curve measurements
        """
        # Extract lane data for easier access
        lane_data = detection_result.data
        detection_status = detection_result.metadata['detection_status']
        
        # Calculate steering angle using our geometric lane analysis
        steering_angle = self.lane_detector._calculate_steering_angle(
            left_curve=lane_data.left_curve,
            right_curve=lane_data.right_curve,
            detection_status=detection_status
        )
        
        # Update vehicle control with both steering and curve information
        # This lets the vehicle controller adjust speed appropriately for curves
        self.vehicle.update_curve_control(
            steering_angle=steering_angle,
            curve_radius=lane_data.curve_radius,
            curve_direction=lane_data.curve_direction
        )
        
        # Update processing metrics if we're tracking them
        if self.latest_metrics is not None:
            self.latest_metrics.steering_angle = steering_angle
    
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