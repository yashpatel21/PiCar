# src/detection/base.py

from abc import ABC, abstractmethod
from utils.types import DetectionFrame, VisualizationResult
from typing import Any, Dict, Generic, TypeVar

T = TypeVar('T')  # Type variable for detector-specific result type

class DetectionResult(Generic[T]):
    """Base class for detection results"""
    def __init__(self, 
                 is_valid: bool,
                 data: T,
                 metadata: Dict[str, Any] = None):
        self.is_valid = is_valid
        self.data = data
        self.metadata = metadata or {}

class Detector(ABC):
    """Base class for all detectors"""
    
    @abstractmethod
    def process_frame(self, frame: DetectionFrame) -> DetectionResult:
        """Process a single frame
        
        Args:
            frame: DetectionFrame containing image data and metadata
            
        Returns:
            DetectionResult containing detection data
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset detector state"""
        pass