# src/viz/lane_viz.py

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from utils.types import ProcessingMetrics, VisualizationResult
from detection.base import DetectionResult
import time

class LaneVisualizer:
    """Creates informative visualizations of lane detection results.
    
    This visualizer provides real-time feedback about the lane detection system's
    performance through multiple complementary views:
    - Main view showing detected lanes and driving metrics
    - Binary threshold view for monitoring lane isolation
    - ROI view for verifying camera coverage
    - Detailed metrics for system performance monitoring
    """
    
    def __init__(self):
        # Configure which visualization elements are enabled
        self.show_binary = True         # Show binary threshold debug view
        self.show_roi = True            # Show region of interest overlay
        self.show_debug = True          # Show additional debug information
        self.show_metrics = True        # Show performance metrics overlay
        
        # Text rendering configuration
        self.text_color = (255, 255, 255)  # White text
        self.text_scale = 0.6              # Medium text size
        self.text_thickness = 2            # Bold text for readability
        self.text_padding = 10             # Spacing between text elements
        
        # Color scheme for different visualization elements (BGR format)
        self.colors = {
            'single_lane': (0, 165, 255),     # Orange for single lane
            'both_lane': (0, 255, 0),         # Green for both lanes
            'center_line': (255, 0, 0),       # Blue for frame center
            'debug_points': (255, 255, 0),    # Cyan for detection points
            'roi_outline': (0, 255, 255)      # Yellow for ROI boundary
        }
        
        # Performance tracking variables
        self.frame_times = []
        self.max_frame_history = 30  # Keep last 30 frames for FPS calculation
        self.last_frame_time = time.time()

    def create_visualization(self, frame: np.ndarray, 
                           detection_result: DetectionResult,
                           debug: bool = True) -> VisualizationResult:
        """Create comprehensive visualization of detection results.
        
        This method generates both the main visualization showing lane detection
        results and additional debug views that help understand the system's
        internal operation.
        
        Args:
            frame: Original BGR frame from camera
            detection_result: Results from lane detection
            debug: Whether to generate additional debug views
            
        Returns:
            VisualizationResult containing all visualization data
        """
        # Calculate FPS
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Update frame times history
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_frame_history:
            self.frame_times.pop(0)
        
        # Calculate average FPS over the last max_frame_history frames
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        # Create main visualization
        main_viz = self._create_main_view(frame, detection_result)
        
        # Create debug views if enabled
        debug_views = {}
        if debug:
            debug_views = self._create_debug_views(frame, detection_result)
        
        # Create metrics
        metrics = self._create_metrics(detection_result)
        
        return VisualizationResult(
            main_frame=main_viz,
            debug_views=debug_views,
            metrics=metrics,
            debug_enabled=debug
        )

    def _create_main_view(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Create main visualization showing lane detection and steering results.
        
        This method renders detection results onto the full camera frame, properly
        accounting for ROI cropping by adjusting coordinate systems. All detected
        points and curves are shifted from ROI coordinates back to full frame
        coordinates for accurate visualization.
        """
        viz_frame = frame.copy()
        
        # Calculate ROI offset - this is crucial for proper visualization
        roi_y_start = int(frame.shape[0] * 0.5)  # 40% from top
        
        # Draw detected lanes with proper vertical offset
        if result.data.left_curve is not None:
            if isinstance(result.data.left_curve, np.ndarray) and len(result.data.left_curve) >= 2:
                # Create offset version of curve points
                offset_curve = result.data.left_curve.copy()
                offset_curve[:, 1] += roi_y_start  # Add offset to y-coordinates
                
                color = (self.colors['single_lane'] 
                        if 'left' in result.metadata['detection_status']
                        else self.colors['both_lane'])
                try:
                    cv2.polylines(viz_frame, [offset_curve], False, color, 2)
                except cv2.error:
                    pass
        
        if result.data.right_curve is not None:
            if isinstance(result.data.right_curve, np.ndarray) and len(result.data.right_curve) >= 2:
                # Create offset version of curve points
                offset_curve = result.data.right_curve.copy()
                offset_curve[:, 1] += roi_y_start  # Add offset to y-coordinates
                
                color = (self.colors['single_lane']
                        if 'right' in result.metadata['detection_status']
                        else self.colors['both_lane'])
                try:
                    cv2.polylines(viz_frame, [offset_curve], False, color, 2)
                except cv2.error:
                    pass
        
        if result.is_valid:
            self._draw_steering_indicator(viz_frame, result)

        self._draw_center_line(viz_frame)
        self._draw_status_overlay(viz_frame, result)
        
        return viz_frame

    def _create_debug_views(self, frame: np.ndarray, 
                        result: DetectionResult) -> Dict[str, np.ndarray]:
        debug_views = {}
        
        # Get dynamic ROI boundaries from metadata
        roi_y_start = result.metadata['roi_y_start']
        roi_y_end = result.metadata['roi_y_end']
        
        # Create binary threshold visualization with proper ROI height
        if self.show_binary and 'binary_frame' in result.metadata:
            binary_viz = cv2.cvtColor(result.metadata['binary_frame'], 
                                    cv2.COLOR_GRAY2BGR)
            debug_views['binary'] = binary_viz
        
        # Create ROI visualization
        if self.show_roi:
            roi_viz = frame.copy()
            roi_vertices = np.array([
                [(0, roi_y_start),
                (0, roi_y_end),
                (frame.shape[1], roi_y_end),
                (frame.shape[1], roi_y_start)]
            ], dtype=np.int32)
            cv2.polylines(roi_viz, [roi_vertices], True, 
                        self.colors['roi_outline'], 2)
            debug_views['roi'] = roi_viz
        
        # Create point detection visualization
        if self.show_debug:
            points_viz = frame.copy()
            
            # Draw left points with dynamic ROI offset
            if result.metadata.get('left_points') is not None:
                for point in result.metadata['left_points']:
                    offset_point = (int(point[0]), 
                                int(point[1] + roi_y_start))
                    cv2.circle(points_viz, offset_point, 2,
                            self.colors['debug_points'], -1)
            
            # Draw right points with dynamic ROI offset
            if result.metadata.get('right_points') is not None:
                for point in result.metadata['right_points']:
                    offset_point = (int(point[0]),
                                int(point[1] + roi_y_start))
                    cv2.circle(points_viz, offset_point, 2,
                            self.colors['debug_points'], -1)
            
            debug_views['points'] = points_viz
        
        return debug_views

    def _create_metrics(self, result: DetectionResult) -> ProcessingMetrics:
        """Create performance metrics from detection results."""
        return ProcessingMetrics(
            frame_time=result.metadata.get('processing_time', 0),
            processing_fps=self.fps,
            detection_status=result.metadata['detection_status'],
            lane_width_confidence=result.metadata['lane_width_confidence'],
            steering_angle=result.data.steering_angle,
            steering_confidence=result.data.steering_confidence,
            current_speed=0.0  # Will be updated when speed control is implemented
        )

    def _draw_center_line(self, frame: np.ndarray):
        """Draw frame center reference line.
        """
        frame_center = frame.shape[1] // 2
        
        # Draw frame center reference line
        cv2.line(frame, 
                (frame_center, frame.shape[0]),
                (frame_center, frame.shape[0]//2),
                self.colors['center_line'], 2)

    def _draw_steering_indicator(self, frame: np.ndarray, result: DetectionResult):
        """Draw visual indicator showing current steering angle.
        
        Creates a vertical line that moves left or right based on the steering angle.
        The line's color changes based on the steering confidence, and its position
        represents the magnitude of the steering angle.
        """
        frame_height, frame_width = frame.shape[:2]
        center_x = frame_width // 2
        
        if result.is_valid:
            # Get steering information from detection results
            steering_angle = result.data.steering_angle
            confidence = result.data.steering_confidence
            
            # Calculate line position based on steering angle
            max_angle = 30.0
            angle_factor = np.clip(steering_angle / max_angle, -1.0, 1.0)
            max_offset = frame_width // 4
            
            # Use cubic function for smoother visual response
            offset = max_offset * (angle_factor ** 3)
            line_x = int(center_x + offset)
            
            # Color varies based on confidence and angle magnitude
            base_color = np.array([0, 255, 0])  # Green base
            if abs(angle_factor) > 0.5:
                base_color = np.array([255, 165, 0])  # Orange for sharp turns
            
            # Fade color based on confidence
            color = tuple(map(int, base_color * confidence))
            
            # Draw steering indicator
            cv2.line(frame, 
                    (line_x, frame.shape[0]),
                    (line_x, frame.shape[0]//2),
                    color, 2)

    def _draw_status_overlay(self, frame: np.ndarray, result: DetectionResult):
        """Draw detection status and steering information overlay.
        
        Shows key information including:
        - Current detection status (which lanes are visible)
        - Steering angle and confidence
        - System performance (FPS)
        """
        y_offset = self.text_padding
        
        # Only show steering angle if detection is valid
        steering_text = (f"Steering: {result.data.steering_angle:+.1f}° "
                        f"({result.data.steering_confidence:.2f})"
                        if result.is_valid else "Steering: N/A")
        
        text_items = [
            f"Detection: {result.metadata['detection_status']}",
            steering_text,
            f"FPS: {self.fps:.1f}"
        ]
        
        for text in text_items:
            self.draw_text(frame, text, (self.text_padding, y_offset + 20))
            y_offset += 25

    def draw_text(self, frame: np.ndarray, text: str, 
                 position: Tuple[int, int],
                 color: Optional[Tuple[int, int, int]] = None):
        """Draw text with consistent styling.
        
        Utility method that ensures all text overlays have consistent
        appearance throughout the visualization.
        """
        cv2.putText(frame, text, position,
                   cv2.FONT_HERSHEY_SIMPLEX, self.text_scale,
                   color or self.text_color, self.text_thickness)