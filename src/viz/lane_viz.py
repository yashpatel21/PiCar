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
        
        # Color scheme for different visualization elements
        self.colors = {
            'detected_lane': (0, 255, 0),     # Green for confirmed lanes
            'estimated_lane': (0, 165, 255),   # Orange for estimated lanes
            'center_line': (0, 0, 255),       # Red for frame center
            'lane_center': (255, 0, 0),       # Blue for lane center
            'debug_points': (0, 255, 255),    # Cyan for detection points
            'roi_outline': (255, 255, 0)      # Yellow for ROI boundary
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

    def _create_main_view(self, frame: np.ndarray, 
                         result: DetectionResult) -> np.ndarray:
        """Create main visualization showing lane detection results.
        
        This view combines the original camera feed with overlays showing:
        - Detected and estimated lane lines
        - Vehicle position relative to lane center
        - Detection status and confidence information
        """
        viz_frame = frame.copy()
        
        # Draw detected lanes with appropriate colors
        if result.data.left_curve is not None:
            # Validate curve points before drawing
            if isinstance(result.data.left_curve, np.ndarray) and len(result.data.left_curve) >= 2:
                # Use different colors for detected vs estimated lanes
                color = (self.colors['detected_lane'] 
                        if 'left' in result.metadata['detection_status']
                        else self.colors['estimated_lane'])
                try:
                    cv2.polylines(viz_frame, [result.data.left_curve], False, color, 2)
                except cv2.error:
                    pass  # Silently handle drawing errors
        
        if result.data.right_curve is not None:
            # Validate curve points before drawing
            if isinstance(result.data.right_curve, np.ndarray) and len(result.data.right_curve) >= 2:
                color = (self.colors['detected_lane']
                        if 'right' in result.metadata['detection_status']
                        else self.colors['estimated_lane'])
                try:
                    cv2.polylines(viz_frame, [result.data.right_curve], False, color, 2)
                except cv2.error:
                    pass  # Silently handle drawing errors
        
        # Draw center lines if we have valid detection
        if result.is_valid:
            self._draw_center_lines(viz_frame, result)
        
        # Add detection status overlay
        self._draw_status_overlay(viz_frame, result)
        
        return viz_frame

    def _create_debug_views(self, frame: np.ndarray, 
                          result: DetectionResult) -> Dict[str, np.ndarray]:
        """Create additional views for system debugging.
        
        Generates several specialized views that help understand and tune
        the detection system:
        - Binary threshold view showing lane isolation
        - ROI view showing detection boundaries
        - Point detection view showing initial lane points
        """
        debug_views = {}
        
        # Create binary threshold visualization
        if self.show_binary and 'binary_frame' in result.metadata:
            binary_viz = cv2.cvtColor(result.metadata['binary_frame'], 
                                    cv2.COLOR_GRAY2BGR)
            # Add threshold value overlay
            threshold_text = f"Threshold: {result.metadata.get('adaptive_threshold', 'N/A')}"
            self.draw_text(binary_viz, threshold_text, (10, 30))
            debug_views['binary'] = binary_viz
        
        # Create ROI visualization
        if self.show_roi:
            roi_viz = frame.copy()
            cv2.polylines(roi_viz, [result.metadata['roi_vertices']], 
                         True, self.colors['roi_outline'], 2)
            debug_views['roi'] = roi_viz
        
        # Create point detection visualization if available
        if self.show_debug and 'left_points' in result.metadata:
            points_viz = frame.copy()
            if result.metadata['left_points'] is not None:
                for point in result.metadata['left_points']:
                    cv2.circle(points_viz, tuple(point), 2, 
                             self.colors['debug_points'], -1)
            if result.metadata['right_points'] is not None:
                for point in result.metadata['right_points']:
                    cv2.circle(points_viz, tuple(point), 2,
                             self.colors['debug_points'], -1)
            debug_views['points'] = points_viz
        
        return debug_views

    def _create_metrics(self, result: DetectionResult) -> ProcessingMetrics:
        """Create performance metrics from detection results.
        
        Combines detection results with system performance data to create
        a comprehensive set of metrics for monitoring system operation.
        """
        return ProcessingMetrics(
            frame_time=result.metadata.get('processing_time', 0),
            center_offset=result.data.center_offset,
            current_speed=result.metadata.get('current_speed', 0),
            steering_angle=result.metadata.get('steering_angle', 0),
            processing_fps=self.fps,
            curve_confidence=1.0 if result.is_valid else 0.0,
            detection_status=result.metadata['detection_status'],
            lane_width_confidence=result.metadata['lane_width_confidence']
        )

    def _draw_center_lines(self, frame: np.ndarray, result: DetectionResult):
        """Draw frame center and lane center reference lines.
        
        These lines help visualize how well the vehicle is centered between
        the detected lanes.
        """
        frame_center = frame.shape[1] // 2
        
        # Draw frame center reference line
        cv2.line(frame, 
                (frame_center, frame.shape[0]),
                (frame_center, frame.shape[0]//2),
                self.colors['center_line'], 2)
        
        # Draw detected lane center line
        if result.data.left_curve is not None and result.data.right_curve is not None:
            lane_center = frame_center + int(result.data.center_offset)
            cv2.line(frame,
                    (lane_center, frame.shape[0]),
                    (lane_center, frame.shape[0]//2),
                    self.colors['lane_center'], 2)

    def _draw_status_overlay(self, frame: np.ndarray, result: DetectionResult):
        """Draw detection status and performance information overlay.
        
        Creates an informative text overlay showing:
        - Which lanes are currently detected
        - Lane width confidence
        - Position offset from lane center
        """
        y_offset = self.text_padding
        text_items = [
            f"Detection: {result.metadata['detection_status']}",
            f"Width Confidence: {result.metadata['lane_width_confidence']:.2f}",
            f"Offset: {result.data.center_offset:+.1f}px",
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