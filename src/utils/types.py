from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, NamedTuple
import numpy as np
from time import time
from abc import abstractmethod

@dataclass 
class DetectionFrame:
   """Holds frame data and metadata for processing.
   
   This class bundles camera frames with important metadata that helps
   track timing and frame dimensions throughout our processing pipeline.
   """
   frame: np.ndarray  # Original BGR frame from camera
   timestamp: float   # Frame capture timestamp in seconds since epoch
   frame_id: int      # Unique sequential identifier for this frame
   width: int        # Frame width in pixels
   height: int       # Frame height in pixels

@dataclass
class LaneDetectionData:
   """Contains results from lane detection processing.
   
   This class represents the core output of lane detection, containing both
   the detected lane features (curves) and derived metrics that inform vehicle
   control decisions. Each field provides key information needed for either
   visualization or vehicle control logic.
   """
   # Core lane detection results
   left_curve: Optional[np.ndarray]   # Points forming left lane curve, None if not detected
   right_curve: Optional[np.ndarray]  # Points forming right lane curve, None if not detected
   
   # Lane position analysis 
   center_offset: float              # Vehicle's offset from lane center in pixels
   curve_radius: Optional[float]     # Radius of current curve in pixels, None if straight
   curve_direction: float           # Direction of curve from -1 (full left) to +1 (full right)
   
   # Steering calculations
   steering_angle: float            # Calculated optimal steering angle in degrees (-30 to +30)
   steering_confidence: float       # Confidence in steering calculation (0 to 1)

@dataclass
class ProcessingMetrics:
    """Tracks performance metrics and detection quality."""
    frame_time: float              # Time to process frame in milliseconds
    processing_fps: float         # Current frames per second processing rate
    
    # Detection quality metrics
    detection_status: str         # Current detection state ('left', 'right', 'both', 'none') 
    lane_width_confidence: float  # Confidence in detected lane width (0 to 1)
    
    # Vehicle control metrics
    steering_angle: float        # Current steering angle in degrees
    steering_confidence: float   # Confidence in steering calculation (0 to 1)  
    current_speed: float        # Current vehicle speed in arbitrary units
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to display-friendly format."""
        return {
            'frame_time': round(self.frame_time, 2),
            'processing_fps': round(self.processing_fps, 1),
            'detection_status': self.detection_status,
            'lane_width_confidence': round(self.lane_width_confidence, 2),
            'steering_angle': round(self.steering_angle, 1),
            'steering_confidence': round(self.steering_confidence, 2),
            'current_speed': round(self.current_speed, 1)
        }
        
@dataclass
class VisualizationResult:
   """Bundles visualization data for display systems.
   
   This class collects all data needed for the web interface and debugging
   displays, including the main visualization frame, additional debug views,
   and current system metrics.
   """
   main_frame: np.ndarray                    # Main annotated video frame
   debug_views: Dict[str, np.ndarray]        # Additional debug visualization frames
   metrics: ProcessingMetrics                # Current system metrics
   debug_enabled: bool = True                # Whether debug views should be shown
   
   def to_dict(self) -> Dict[str, Any]:
       """Convert visualization data to web-friendly format."""
       return {
           'metrics': self.metrics.to_dict(),
           'debug_enabled': self.debug_enabled
       }

@dataclass
class DetectionResult:
   """Contains complete detection results and metadata.
   
   This class is the primary output of the lane detection system, containing
   both the core detection results and additional metadata useful for debugging
   and visualization. It provides convenience methods for accessing commonly
   needed values.
   """
   is_valid: bool                # Whether detection was successful
   data: LaneDetectionData      # Core detection results including steering
   metadata: Dict[str, Any]     # Additional debugging and visualization data

class Detector:
   """Base class defining the interface for all detectors in the system.
   
   This abstract base class ensures all detectors provide the basic
   functionality needed by the rest of the system.
   """
   
   @abstractmethod
   def process_frame(self, frame: DetectionFrame) -> DetectionResult:
       """Process a single frame and return detection results."""
       pass

   @abstractmethod
   def reset(self):
       """Reset detector state and clear any cached data."""
       pass

   @abstractmethod
   def get_config(self) -> Dict[str, Any]:
       """Get current configuration parameters."""
       pass