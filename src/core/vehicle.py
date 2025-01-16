# src/core/vehicle.py

from picarx import Picarx
import time
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class VehicleState:
    """Maintains the current state of all vehicle components."""
    speed: float = 0.0           # Current speed (-100 to 100)
    steering_angle: float = 0.0  # Current steering angle (-30 to 30)
    camera_tilt: float = 0.0     # Current camera tilt angle (-30 to 30)
    camera_pan: float = 0.0      # Current camera pan angle (-30 to 30)
    is_moving: bool = False
    curve_radius: Optional[float] = None  # Radius of current curve (if any)
    lateral_acceleration: float = 0.0     # Side-to-side acceleration

class VehicleController:
    """Controls vehicle movement with sophisticated curve handling and smooth transitions."""
    
    def __init__(self):
        """Initialize vehicle controller with carefully tuned parameters."""
        # Initialize PiCar-X hardware interface
        self.px = Picarx()
        self.state = VehicleState()
        
        # Movement safety parameters
        self.max_speed = 50.0            # Maximum allowed speed
        self.max_steering_angle = 30.0   # Maximum steering angle
        self.stopping_time = 0.5         # Time to come to complete stop
        
        # Camera parameters
        self.default_camera_tilt = -5.0  # Optimal downward angle for lane viewing
        self.default_camera_pan = 7.0     # Slightly right of center for better perspective
        self.max_camera_angle = 30.0      # Maximum camera angle in any direction
        
        # Enhanced curve handling parameters
        self.min_curve_speed = 20.0       # Minimum speed in sharpest turns
        self.max_lateral_accel = 0.3      # Maximum allowed lateral acceleration
        self.steering_smoothing = 0.7     # Primary steering smoothing factor
        self.prev_steering_angle = 0.0    # For smooth steering transitions
        
        # Additional steering smoothing parameters
        self.steering_history = []
        self.steering_history_size = 3    # Reduced from 5 to maintain responsiveness
        
        # Perform initial setup
        self.initialize_vehicle()
    
    def initialize_vehicle(self):
        """Set up vehicle in a safe starting state."""
        self.stop(gradual=False)
        self.set_steering_angle(0)
        self.initialize_camera()
        time.sleep(0.5)
    
    def initialize_camera(self):
        """Position camera for optimal lane detection."""
        self.set_camera_pan(self.default_camera_pan)
        self.set_camera_tilt(self.default_camera_tilt)
        time.sleep(0.5)
    
    def set_camera_tilt(self, angle: float) -> None:
        """Adjust camera tilt with safety limits."""
        safe_angle = max(min(angle, self.max_camera_angle), -self.max_camera_angle)
        self.px.set_cam_tilt_angle(safe_angle)
        self.state.camera_tilt = safe_angle
    
    def set_camera_pan(self, angle: float) -> None:
        """Adjust camera pan with safety limits."""
        safe_angle = max(min(angle, self.max_camera_angle), -self.max_camera_angle)
        self.px.set_cam_pan_angle(safe_angle)
        self.state.camera_pan = safe_angle
    
    def calculate_curve_speed(self, curve_radius: Optional[float], 
                            curve_direction: float) -> float:
        """Calculate safe speed based on curve characteristics."""
        if curve_radius is None or curve_radius > 1000:  # Effectively straight
            return self.max_speed
        
        # Calculate safe speed using physics-based approach
        safe_speed = np.sqrt(curve_radius * self.max_lateral_accel)
        
        # Adjust for curve direction severity
        direction_factor = 1.0 - abs(curve_direction) * 0.5
        adjusted_speed = safe_speed * direction_factor
        
        return max(min(adjusted_speed, self.max_speed), self.min_curve_speed)
    
    def set_steering_angle(self, angle: float, smooth: bool = True) -> None:
        """Set steering angle with sophisticated smoothing."""
        # Apply safety limits
        safe_angle = max(min(angle, self.max_steering_angle), -self.max_steering_angle)
        
        if smooth:
            # Add to history for additional smoothing
            self.steering_history.append(safe_angle)
            if len(self.steering_history) > self.steering_history_size:
                self.steering_history.pop(0)
            
            # Calculate moving average
            avg_angle = sum(self.steering_history) / len(self.steering_history)
            
            # Apply primary exponential smoothing
            smoothed_angle = (self.steering_smoothing * self.prev_steering_angle +
                            (1 - self.steering_smoothing) * avg_angle)
            
            self.prev_steering_angle = smoothed_angle
            safe_angle = smoothed_angle
        
        self.px.set_dir_servo_angle(safe_angle)
        self.state.steering_angle = safe_angle
    
    def set_speed(self, speed: float) -> None:
        """Set vehicle speed with safety limits."""
        safe_speed = max(min(speed, self.max_speed), -self.max_speed)
        self.state.speed = safe_speed
        self.state.is_moving = safe_speed != 0
        self.px.forward(safe_speed)
    
    def stop(self, gradual: bool = True) -> None:
        """Stop vehicle with optional gradual deceleration."""
        if gradual and self.state.speed != 0:
            initial_speed = self.state.speed
            steps = 10
            for i in range(steps):
                reduced_speed = initial_speed * (steps - i - 1) / steps
                self.set_speed(reduced_speed)
                time.sleep(self.stopping_time / steps)
        
        self.set_speed(0)
        self.state.is_moving = False
    
    def update_curve_control(self, steering_angle: float, 
                           curve_radius: Optional[float] = None,
                           curve_direction: float = 0.0) -> None:
        """Update vehicle control based on lane detection results.
        
        This method combines steering angle setting with speed control,
        ensuring appropriate speed adjustments for curves while maintaining
        smooth steering transitions.
        
        Args:
            steering_angle: Desired steering angle from lane detection
            curve_radius: Detected curve radius (if any)
            curve_direction: Overall curve direction (-1 to 1, 0 is straight)
        """
        # Apply steering with built-in smoothing
        self.set_steering_angle(steering_angle, smooth=True)
        
        # Update speed based on curve characteristics
        safe_speed = self.calculate_curve_speed(curve_radius, curve_direction)
        self.set_speed(safe_speed)
    
    def get_state(self) -> VehicleState:
        """Get current vehicle state."""
        return self.state
    
    def cleanup(self):
        """Clean shutdown of vehicle systems."""
        self.stop(gradual=False)
        self.set_steering_angle(0, smooth=False)
        self.set_camera_pan(0)
        self.set_camera_tilt(0)
        time.sleep(0.5)