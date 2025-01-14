# src/viz/base.py

from abc import ABC, abstractmethod
from utils.types import VisualizationResult
from typing import Any, Dict, Optional
import numpy as np
import cv2

class Visualizer(ABC):
    """Base class for visualization handlers"""
    
    def __init__(self):
        self.enabled_layers = set()  # Set of enabled visualization layers
        self.debug_enabled = False
        
    @abstractmethod
    def create_visualization(self, 
                           frame: np.ndarray,
                           detection_result: Any,
                           debug: bool = False) -> VisualizationResult:
        """Create visualization from detection results
        
        Args:
            frame: Original BGR frame
            detection_result: Detector-specific results
            debug: Whether to include debug visualizations
            
        Returns:
            VisualizationResult containing all visualization data
        """
        pass
    
    def enable_layer(self, layer_name: str):
        """Enable a specific visualization layer"""
        self.enabled_layers.add(layer_name)
        
    def disable_layer(self, layer_name: str):
        """Disable a specific visualization layer"""
        self.enabled_layers.discard(layer_name)
        
    def toggle_debug(self, enabled: bool):
        """Toggle debug visualization mode"""
        self.debug_enabled = enabled

    def draw_text(self, 
                 frame: np.ndarray,
                 text: str,
                 position: tuple,
                 color: tuple = (255, 255, 255),
                 scale: float = 0.6,
                 thickness: int = 2):
        """Utility method to draw text with consistent styling"""
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                   scale, color, thickness)
    
    def draw_metrics(self, 
                    frame: np.ndarray,
                    metrics: Dict[str, Any],
                    start_y: int = 30,
                    spacing: int = 25):
        """Utility method to draw metrics overlay"""
        if 'metrics' not in self.enabled_layers:
            return
            
        y = start_y
        for key, value in metrics.items():
            text = f"{key}: {value}"
            self.draw_text(frame, text, (10, y))
            y += spacing