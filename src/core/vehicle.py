# src/core/vehicle.py

from picarx import Picarx
import time
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class VehicleState:
    """Maintains the current state of all vehicle components.
    
    This class serves as a central source of truth for the vehicle's configuration,
    tracking everything from basic movement to camera positioning. Having all state
    in one place makes it easier to monitor and debug the system's behavior.
    
    Attributes:
        speed: Current speed (-100 to 100, negative is backward)
        steering_angle: Current wheel angle (-30 to 30 degrees)
        camera_tilt: Current camera tilt angle (-30 to 30 degrees)
        camera_pan: Current camera pan angle (-30 to 30 degrees)
        is_moving: Whether the vehicle is currently in motion
        curve_radius: Current curve radius being navigated (if any)
        lateral_acceleration: Side-to-side acceleration in curves
    """
    speed: float = 0.0           # Current speed (-100 to 100)
    steering_angle: float = 0.0  # Current steering angle (-30 to 30)
    camera_tilt: float = 0.0     # Current camera tilt angle (-30 to 30)
    camera_pan: float = 0.0      # Current camera pan angle (-30 to 30)
    is_moving: bool = False
    curve_radius: Optional[float] = None  # Radius of current curve (if any)
    lateral_acceleration: float = 0.0     # Side-to-side acceleration

class VehicleController:
    """Controls all physical aspects of the vehicle including movement and camera.
    
    This class provides a high-level interface for controlling the PiCar-X. It ensures
    safe operation by implementing limits on speed and steering, smooth transitions
    between states, and proper camera positioning for lane detection.
    """
    
    def __init__(self):
        """Initialize vehicle controller with safe default settings."""
        # Initialize PiCar-X hardware interface
        self.px = Picarx()
        self.state = VehicleState()
        
        # Movement safety parameters
        self.max_speed = 50.0        # Maximum allowed speed
        self.max_steering_angle = 30.0  # Maximum steering angle
        self.stopping_time = 0.5     # Time to come to complete stop
        
        # Camera parameters
        self.default_camera_tilt = -20.0  # Default downward angle for optimal view
        self.default_camera_pan = 7.0     # Center position
        self.max_camera_angle = 30.0      # Maximum camera angle in any direction
        
        # Enhanced curve handling parameters
        self.min_curve_speed = 20.0      # Minimum speed in sharpest turns
        self.max_lateral_accel = 0.3     # Maximum allowed lateral acceleration
        self.steering_smoothing = 0.7    # Steering angle smoothing factor
        self.prev_steering_angle = 0.0   # For smooth steering transitions
        
        # Perform initial setup
        self.initialize_vehicle()
    
    def initialize_vehicle(self):
        """Set up vehicle in a known safe state with proper camera positioning.
        
        This method ensures the vehicle starts in a consistent state, which is
        crucial for reliable operation. It stops any movement, centers the wheels,
        and positions the camera optimally for lane detection.
        """
        # Start with vehicle stopped and wheels straight
        self.stop(gradual=False)
        self.set_steering_angle(0)
        
        # Initialize camera position
        self.initialize_camera()
        
        # Allow servos time to reach position
        time.sleep(0.5)
    
    def initialize_camera(self):
        """Position camera for optimal lane and sign detection.
        
        The camera needs to be positioned to see both the immediate lane area
        and any upcoming curves or signs. The default tilt angle provides a
        good balance between these needs.
        """
        # Set camera to centered pan position
        self.set_camera_pan(self.default_camera_pan)
        
        # Set optimal downward tilt
        self.set_camera_tilt(self.default_camera_tilt)
        
        # Allow servos time to reach position
        time.sleep(0.5)
    
    def set_camera_tilt(self, angle: float) -> None:
        """Adjust the camera's tilt angle (up/down).
        
        Args:
            angle: Desired tilt angle in degrees (-30 to 30)
                  Negative angles point camera down, positive angles point up
        """
        safe_angle = max(min(angle, self.max_camera_angle), -self.max_camera_angle)
        self.px.set_cam_tilt_angle(safe_angle)
        self.state.camera_tilt = safe_angle
    
    def set_camera_pan(self, angle: float) -> None:
        """Adjust the camera's pan angle (left/right).
        
        Args:
            angle: Desired pan angle in degrees (-30 to 30)
                  Negative angles pan left, positive angles pan right
        """
        safe_angle = max(min(angle, self.max_camera_angle), -self.max_camera_angle)
        self.px.set_cam_pan_angle(safe_angle)
        self.state.camera_pan = safe_angle
    
    def calculate_curve_speed(self, curve_radius: Optional[float], 
                            curve_direction: float) -> float:
        """Calculate safe speed based on curve characteristics.
        
        This method implements speed control for curves, ensuring the car slows
        down appropriately for sharp turns while maintaining higher speeds on
        gentle curves or straight sections.
        
        Args:
            curve_radius: Radius of the curve in pixels (None if straight)
            curve_direction: Direction of curve (-1 to 1, 0 is straight)
            
        Returns:
            Safe speed for current curve conditions
        """
        if curve_radius is None or curve_radius > 1000:  # Effectively straight
            return self.max_speed
            
        # Calculate safe speed based on curve radius and lateral acceleration
        # Using the formula v = sqrt(r * a) where r is radius and a is acceleration
        safe_speed = np.sqrt(curve_radius * self.max_lateral_accel)
        
        # Scale speed based on curve direction severity
        direction_factor = 1.0 - abs(curve_direction) * 0.5
        adjusted_speed = safe_speed * direction_factor
        
        # Ensure speed stays within safe limits
        return max(min(adjusted_speed, self.max_speed), self.min_curve_speed)
    
    def set_steering_angle(self, angle: float, smooth: bool = True) -> None:
        """Set vehicle steering angle with smooth transitions.
        
        Args:
            angle: Desired steering angle (-30 to 30 degrees)
            smooth: Whether to apply steering smoothing
        """
        # Apply safety limits
        safe_angle = max(min(angle, self.max_steering_angle), -self.max_steering_angle)
        
        if smooth:
            # Apply exponential smoothing to steering angle for smoother transitions
            smoothed_angle = (self.steering_smoothing * self.prev_steering_angle +
                            (1 - self.steering_smoothing) * safe_angle)
            self.prev_steering_angle = smoothed_angle
            safe_angle = smoothed_angle
        
        self.px.set_dir_servo_angle(safe_angle)
        self.state.steering_angle = safe_angle
    
    def set_speed(self, speed: float) -> None:
        """Set vehicle speed with safety limits.
        
        Args:
            speed: Desired speed (-100 to 100)
                  Negative values for reverse, positive for forward
        """
        safe_speed = max(min(speed, self.max_speed), -self.max_speed)
        
        # Update state
        self.state.speed = safe_speed
        self.state.is_moving = safe_speed != 0
        
        # Control vehicle movement
        self.px.forward(safe_speed)  # forward() handles both positive and negative speeds
    
    def stop(self, gradual: bool = True) -> None:
        """Stop vehicle movement with optional gradual deceleration.
        
        Args:
            gradual: If True, gradually slow to stop; if False, stop immediately.
                    Gradual stopping provides smoother operation but takes longer.
        """
        if gradual and self.state.speed != 0:
            # Implement gradual speed reduction
            initial_speed = self.state.speed
            steps = 10
            for i in range(steps):
                reduced_speed = initial_speed * (steps - i - 1) / steps
                self.set_speed(reduced_speed)
                time.sleep(self.stopping_time / steps)
        
        # Ensure complete stop
        self.set_speed(0)
        self.state.is_moving = False
    
    def emergency_stop(self) -> None:
        """Immediately stop the vehicle in emergency situations.
        
        This method provides the fastest possible stopping response by:
        1. Immediately cutting power to motors
        2. Bypassing gradual speed reduction
        3. Ensuring vehicle state is updated
        """
        self.stop(gradual=False)
    
    def get_state(self) -> VehicleState:
        """Get the current state of the vehicle.
        
        Returns:
            VehicleState object containing current speed, angles, and status
        """
        return self.state

    def cleanup(self):
        """Perform cleanup when shutting down the vehicle controller.
        
        This method ensures the vehicle is left in a safe state by:
        1. Stopping all movement
        2. Centering wheels and camera
        3. Allowing time for servos to reach position
        """
        self.stop(gradual=False)
        self.set_steering_angle(0, smooth=False)
        self.set_camera_pan(0)
        self.set_camera_tilt(0)
        time.sleep(0.5)