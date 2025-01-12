from picarx import Picarx
import time
from typing import Optional
from dataclasses import dataclass

@dataclass
class VehicleState:
    """Current state of the vehicle."""
    speed: float = 0.0        # Current speed (-100 to 100)
    steering_angle: float = 0.0  # Current steering angle (-30 to 30)
    is_moving: bool = False

class VehicleController:
    """Manages vehicle movement and state.
    
    This class provides high-level control over the vehicle's movement,
    including speed control, steering, and safety features.
    """
    
    def __init__(self):
        """Initialize vehicle controller and safety parameters."""
        self.px = Picarx()
        self.state = VehicleState()
        
        # Safety parameters
        self.max_speed = 50.0  # Maximum allowed speed
        self.max_steering_angle = 30.0  # Maximum steering angle
        self.stopping_time = 0.5  # Time to come to complete stop
        
    def set_steering_angle(self, angle: float) -> None:
        """Set vehicle steering angle with safety limits.
        
        Args:
            angle: Desired steering angle in degrees (-30 to 30)
        """
        # Apply safety limits
        safe_angle = max(min(angle, self.max_steering_angle), 
                        -self.max_steering_angle)
        
        self.state.steering_angle = safe_angle
        self.px.set_dir_servo_angle(safe_angle)
        
    def set_speed(self, speed: float) -> None:
        """Set vehicle speed with safety limits.
        
        Args:
            speed: Desired speed (-100 to 100)
        """
        # Apply safety limits
        safe_speed = max(min(speed, self.max_speed), -self.max_speed)
        
        self.state.speed = safe_speed
        self.state.is_moving = safe_speed != 0
        
        if safe_speed >= 0:
            self.px.forward(safe_speed)
        else:
            self.px.backward(abs(safe_speed))
            
    def stop(self, gradual: bool = True) -> None:
        """Stop the vehicle.
        
        Args:
            gradual: If True, gradually slow to stop; if False, stop immediately
        """
        if gradual and self.state.speed != 0:
            # Gradually reduce speed
            initial_speed = self.state.speed
            steps = 10
            for i in range(steps):
                reduced_speed = initial_speed * (steps - i - 1) / steps
                self.set_speed(reduced_speed)
                time.sleep(self.stopping_time / steps)
        
        self.set_speed(0)
        self.state.is_moving = False
        
    def emergency_stop(self) -> None:
        """Immediate stop for emergency situations."""
        self.stop(gradual=False)