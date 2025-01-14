# src/utils/types.py

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
    timestamp: float   # Frame capture timestamp
    frame_id: int      # Unique frame identifier
    width: int        # Frame width in pixels
    height: int       # Frame height in pixels

@dataclass
class LaneDetectionData:
    """Contains results from lane detection processing.
    
    This class has been enhanced to better handle partial lane detection
    scenarios and provide more detailed information about detected lanes.
    """
    left_curve: Optional[np.ndarray]   # Left lane curve points
    right_curve: Optional[np.ndarray]  # Right lane curve points
    center_offset: float              # Offset from lane center
    curve_radius: Optional[float]     # Curve radius if turning
    curve_direction: float           # Direction of curve (-1 left to 1 right)

@dataclass
class ProcessingMetrics:
    """Tracks performance metrics and detection quality.
    
    This enhanced version includes additional metrics to help monitor
    lane detection quality and system performance.
    """
    frame_time: float              # Processing time in milliseconds
    center_offset: float          # Lane center offset
    current_speed: float          # Vehicle speed
    steering_angle: float         # Current steering angle
    processing_fps: float         # Frames per second
    curve_confidence: float       # Lane detection confidence
    detection_status: str         # Which lanes are visible
    lane_width_confidence: float  # Confidence in width estimate

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for web display."""
        return {
            'frame_time': round(self.frame_time, 2),
            'center_offset': round(self.center_offset, 2),
            'current_speed': round(self.current_speed, 2),
            'steering_angle': round(self.steering_angle, 2),
            'processing_fps': round(self.processing_fps, 1),
            'curve_confidence': round(self.curve_confidence, 3),
            'detection_status': self.detection_status,
            'lane_width_confidence': round(self.lane_width_confidence, 2)
        }

@dataclass
class VisualizationResult:
    """Bundles visualization data for display systems.
    
    This class has been enhanced to support additional debug views
    and provide more detailed information about the detection state.
    """
    main_frame: np.ndarray                    # Main visualization frame
    debug_views: Dict[str, np.ndarray]        # Named debug views
    metrics: ProcessingMetrics                # Current metrics
    debug_enabled: bool = True                # Debug view toggle
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert visualization data to web-friendly format."""
        return {
            'metrics': self.metrics.to_dict(),
            'debug_enabled': self.debug_enabled
        }

@dataclass
class DetectionResult:
    """Contains complete detection results and metadata.
    
    This class wraps detection data with validity information and
    debugging metadata to help monitor and tune the system.
    """
    is_valid: bool                # Whether detection succeeded
    data: LaneDetectionData      # Detection results
    metadata: Dict[str, Any]     # Additional information and debug data

class DetectorBase:
    """Base class defining the interface for all detectors."""
    
    @abstractmethod
    def process_frame(self, frame: DetectionFrame) -> DetectionResult:
        """Process a single frame and return detection results."""
        pass

    @abstractmethod
    def reset(self):
        """Reset detector state."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration parameters."""
        pass