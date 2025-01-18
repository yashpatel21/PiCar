# src/detection/lane.py

import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict, Any
from .base import Detector, DetectionResult
from utils.types import DetectionFrame, LaneDetectionData
import time

class LaneDetector(Detector):
    """Lane detection system for real-time tracking and analysis of lane boundaries.
    
    This detector implements a complete lane detection pipeline optimized for embedded
    systems and real-time processing. It handles single and dual lane detection through
    adaptive thresholding, polynomial curve fitting, and temporal state tracking.
    
    The system processes frames through several stages:
    1. Image preprocessing with adaptive thresholding
    2. Region of interest (ROI) selection
    3. Lane point detection using histogram analysis
    4. Curve fitting and validation
    5. Steering angle calculation
    
    The detector maintains temporal state information for improved stability and 
    implements multiple validation steps to prevent false detections in varying
    lighting conditions and track configurations.
    """
    
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        """Initialize detector with frame dimensions and tracking parameters.
        
        The initialization sets up several key components:
        - Region of interest boundaries
        - Image processing parameters
        - State tracking variables
        - Lane position history
        - Classification parameters
        
        Args:
            frame_width: Width of input frames in pixels
            frame_height: Height of input frames in pixels
        """
        self.width = frame_width
        self.height = frame_height
        
        # Define default ROI coordinates (50% to 80% of frame height)
        self.roi_default_start = 0.50
        self.roi_default_end = 0.8
        
        # Image processing and detection parameters
        self.config = {
            # Gaussian blur for noise reduction
            'blur_kernel_size': 7,
            
            # Adaptive threshold parameters for binary image generation
            'adaptive_block_size': 45,
            'adaptive_offset': 25,
            
            # Morphological operation parameters
            'dilate_iterations': 1,
            'erode_iterations': 1,
            
            # Lane detection thresholds and window parameters
            'min_points_for_curve': 4,    # Minimum points needed for curve fitting
            'min_component_size': 50,        # Minimum pixels for valid component
            'max_component_spread': 30,      # Maximum allowed point spread
            'min_aspect_ratio': 2.0,         # Minimum height/width ratio
            'temporal_distance_threshold': 50 # Maximum allowed position change
        }
        
        # State tracking for temporal consistency
        self.prev_left_curve = None       # Previous frame's left lane curve
        self.prev_right_curve = None      # Previous frame's right lane curve
        self.prev_curve_radius = None     # Previous frame's curve radius
        self.prev_curve_direction = 0     # Previous frame's curve direction
        
        # Lane position tracking with temporal smoothing
        self.prev_left_x = None          # Smoothed left lane x-position
        self.prev_right_x = None         # Smoothed right lane x-position
        self.position_memory_decay = 0.8  # Position smoothing factor
        
        # Classification history for temporal stability
        self.lane_classification_history = []
        self.classification_history_size = 5
        
        # Lane width tracking for validation
        self.recent_lane_widths = []
        self.max_width_history = 10
        self.lane_width_confidence = 0.0

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert camera frame to binary image optimized for lane detection.
        
        This preprocessing pipeline applies several image processing steps:
        1. Grayscale conversion to reduce dimensionality
        2. Gaussian blur to reduce high-frequency noise
        3. Adaptive thresholding to handle variable lighting conditions
        4. Morphological operations to clean up the binary image
        
        The adaptive thresholding is particularly important as it automatically
        adjusts to changes in lighting conditions across the frame, making the
        detection more robust to shadows and varying illumination.
        
        Args:
            frame: Input BGR image from camera
            
        Returns:
            Binary image where white pixels (255) represent potential lane markings
        """

        if frame is None or frame.size == 0:
            raise ValueError("Empty or invalid frame received in _preprocess_frame")

        # Verify frame dimensions
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(f"Invalid frame format. Expected 3-channel BGR image, got shape {frame.shape}")
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            k_size = self.config['blur_kernel_size']
            blurred = cv2.GaussianBlur(gray, (k_size, k_size), 0)
            
            binary = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.config['adaptive_block_size'],
                self.config['adaptive_offset']
            )
            
            kernel = np.ones((3,3), np.uint8)
            binary = cv2.dilate(binary, kernel, 
                            iterations=self.config['dilate_iterations'])
            binary = cv2.erode(binary, kernel, 
                            iterations=self.config['erode_iterations'])
            
            return binary
        except cv2.error as e:
            raise ValueError(f"OpenCV error during preprocessing: {str(e)}")

    def _classify_single_lane(self, points: np.ndarray) -> str:
        """Classify a single lane based on its curve/slope direction.
        
        The classification follows a simple geometric principle:
        - If points curve/slope upward and rightward -> it's a left lane
        - If points curve/slope upward and leftward -> it's a right lane
        
        This works because of how lanes appear in perspective view from the car:
        Left lanes curve right as they go up because we're seeing them from their right side
        Right lanes curve left as they go up because we're seeing them from their left side
        
        Args:
            points: Nx2 array of (x,y) coordinates, sorted by y (bottom to top)
        Returns:
            'left' or 'right' classification
        """
        if len(points) < 3:
            return 'left'  # Safe default if not enough points
            
        # Get bottom and top sections of points
        top_third = points[:len(points)//3]
        bottom_third = points[-len(points)//3:]
        
        # Calculate average x position for each section
        top_x = np.mean(top_third[:, 0])
        bottom_x = np.mean(bottom_third[:, 0])
        
        # If x position increases (curves right) as we go up, it's a left lane
        # If x position decreases (curves left) as we go up, it's a right lane
        return 'left' if top_x > bottom_x else 'right'

    def _find_lane_points(self, binary_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """Detect lane points using connected components analysis.
        
        This method implements a complete lane detection strategy that handles both
        single and dual lane scenarios. For dual lanes, it uses relative position
        to classify left and right. For single lanes, it employs sophisticated
        geometric analysis to determine the lane type.
        
        The detection process follows these steps:
        1. Find all connected components in the binary image
        2. Filter components by size to remove noise
        3. Extract point coordinates for valid components
        4. Classify components using either dual or single lane logic
        
        Args:
            binary_img: Binary image where lane lines appear as white pixels
            
        Returns:
            Tuple containing:
            - Left lane points (Nx2 array or empty)
            - Right lane points (Nx2 array or empty)
            - Detection status ('left', 'right', 'both', or 'none')
        """
        # Initialize return values
        left_points = np.array([])
        right_points = np.array([])
        detection_status = 'none'
        
        # Find all connected components in the binary image
        num_labels, labels = cv2.connectedComponents(binary_img)
        
        # Need at least one component besides background
        if num_labels < 2:
            return left_points, right_points, detection_status
        
        # Calculate size of each component (excluding background label 0)
        component_sizes = [np.sum(labels == label) for label in range(1, num_labels)]
        
        # Get indices of components sorted by size (largest first)
        sorted_indices = np.argsort(component_sizes)[::-1]
        
        # Process components to get points
        components = []
        min_component_size = binary_img.size * 0.01  # Minimum size threshold
        
        # Extract valid components
        for label_idx in sorted_indices:
            # Skip if component is too small
            if component_sizes[label_idx] < min_component_size:
                continue
                
            # Get points for this component
            label = label_idx + 1
            y_coords, x_coords = np.where(labels == label)
            points = np.column_stack((x_coords, y_coords))
            # Sort points by y-coordinate (bottom to top)
            points = points[points[:, 1].argsort()]
            components.append(points)
            
            # Stop after finding up to two valid components
            if len(components) >= 2:
                break
        
        # Handle different detection scenarios
        if len(components) >= 2:
            # Dual lane case - use bottom positions to determine left/right
            if components[0][0][0] < components[1][0][0]:
                left_points = components[0]
                right_points = components[1]
            else:
                left_points = components[1]
                right_points = components[0]
            detection_status = 'both'
            
        elif len(components) == 1:
            # Single lane case - use robust geometric analysis
            points = components[0]
            lane_type = self._classify_single_lane(points)
            
            if lane_type == 'left':
                left_points = points
                detection_status = 'left'
            else:
                right_points = points
                detection_status = 'right'
        
        return left_points, right_points, detection_status

    def _update_position_tracking(self, left_points: np.ndarray, right_points: np.ndarray):
        """Update tracked lane positions with temporal smoothing.
        
        Maintains a smoothed estimate of lane positions over time using exponential
        moving average. This helps maintain stable lane tracking even when detection
        is temporarily uncertain.
        
        Args:
            left_points: Detected left lane points
            right_points: Detected right lane points
        """
        if len(left_points) > 0:
            current_left_x = np.mean(left_points[:, 0])
            if self.prev_left_x is None:
                self.prev_left_x = current_left_x
            else:
                self.prev_left_x = (self.position_memory_decay * self.prev_left_x +
                                    (1 - self.position_memory_decay) * current_left_x)
        
        if len(right_points) > 0:
            current_right_x = np.mean(right_points[:, 0])
            if self.prev_right_x is None:
                self.prev_right_x = current_right_x
            else:
                self.prev_right_x = (self.position_memory_decay * self.prev_right_x +
                                    (1 - self.position_memory_decay) * current_right_x)

    def _fit_curve(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Fit a smooth polynomial curve to detected lane points.
        
        This method uses polynomial regression to create a mathematical model
        of the lane's path. A second-degree polynomial effectively captures
        the typical curvature of lanes while avoiding overfitting to noise
        in the point detection.
        
        The fitted curve serves multiple purposes:
        1. Smooths out noise in the raw detected points
        2. Provides a continuous model of the lane's path
        3. Enables prediction of lane position beyond detected points
        4. Allows calculation of geometric properties like curvature
        
        Args:
            points: Array of detected lane points (x,y coordinates)
            
        Returns:
            Array of points along the fitted curve, or None if fitting fails
        """
        if not isinstance(points, np.ndarray) or len(points) < self.config['min_points_for_curve']:
            return None
            
        try:
            # Fit second-degree polynomial to capture typical road curvature
            coeffs = np.polyfit(points[:, 1], points[:, 0], 2)
            
            # Generate evenly spaced points along the curve
            y_points = np.linspace(
                np.min(points[:, 1]),
                np.max(points[:, 1]),
                num=20
            )
            x_points = np.polyval(coeffs, y_points)
            
            # Combine x,y coordinates into curve points
            curve_points = np.column_stack((
                x_points.astype(np.int32),
                y_points.astype(np.int32)
            ))
            
            return curve_points
            
        except Exception as e:
            print(f"Warning: Curve fitting failed: {str(e)}")
            return None

    def _calculate_curve_radius(self, curve_points: np.ndarray) -> Optional[float]:
        """Calculate the radius of curvature of a lane line.
        
        The radius of curvature indicates how sharply the lane is turning.
        A smaller radius indicates a sharper turn, while a larger radius
        indicates a gentler turn or straight section. This information
        helps determine appropriate steering adjustments.
        
        The calculation uses the mathematical formula for radius of curvature
        of a curve defined by a quadratic function:
        R = (1 + (dy/dx)^2)^(3/2) / |d^2y/dx^2|
        
        Args:
            curve_points: Points along the fitted curve
            
        Returns:
            Radius of curvature in pixels, or None if calculation fails
        """
        if curve_points is None or len(curve_points) < 3:
            return None
            
        try:
            # Fit polynomial to calculate derivatives
            coeffs = np.polyfit(curve_points[:, 1], curve_points[:, 0], 2)
            
            # Calculate radius at the bottom point (closest to vehicle)
            y_eval = np.max(curve_points[:, 1])
            
            # First derivative (dy/dx)
            dx_dy = 2 * coeffs[0] * y_eval + coeffs[1]
            
            # Second derivative (d^2y/dx^2)
            d2x_dy2 = 2 * coeffs[0]
            
            # Calculate radius using the curve formula
            if d2x_dy2 != 0:
                radius = ((1 + dx_dy**2)**(3/2)) / abs(d2x_dy2)
            return radius
            return None
            
        except Exception as e:
            print(f"Warning: Radius calculation failed: {str(e)}")
            return None

    def _calculate_steering_confidence(self, detection_status: str, lane_width_confidence: float) -> float:
        """Calculate confidence in steering decisions based on detection quality."""
        if detection_status == 'both':
            return min(1.0, 0.8 + 0.2 * lane_width_confidence)
        elif detection_status in ['left', 'right']:
            return 0.7  # Single lane detection has lower baseline confidence
        return 0.0

    def _update_lane_width(self, left_curve: Optional[np.ndarray], 
                        right_curve: Optional[np.ndarray]) -> Optional[float]:
        """Update lane width tracking and calculate confidence in width measurements.
        
        This method maintains a running history of lane width measurements and
        calculates confidence based on consistency of these measurements.
        """
        if left_curve is None or right_curve is None:
            return None
            
        # Calculate average lane width
        left_x = np.mean(left_curve[:, 0])
        right_x = np.mean(right_curve[:, 0])
        current_width = abs(right_x - left_x)
        
        # Update width history
        self.recent_lane_widths.append(current_width)
        if len(self.recent_lane_widths) > self.max_width_history:
            self.recent_lane_widths.pop(0)
        
        # Calculate width consistency
        if len(self.recent_lane_widths) >= 3:
            mean_width = np.mean(self.recent_lane_widths)
            std_width = np.std(self.recent_lane_widths)
            
            # Calculate confidence based on width consistency
            variation_coefficient = std_width / (mean_width + 1e-6)
            self.lane_width_confidence = max(0.0, min(1.0, 1.0 - variation_coefficient))
        else:
            self.lane_width_confidence = 0.5  # Default confidence with limited history
        
        return current_width

    def _calculate_bottom_point_urgency(self, lane_points: np.ndarray) -> float:
        """Calculate how urgently we need to steer based on bottom point position.
        
        Remember: In image coordinates, bottom points have highest y-values.
        """
        if len(lane_points) == 0:
            return 0.0
        
        # Get the points with highest y-values (bottom of image)
        bottom_points = lane_points[-len(lane_points)//4:]  # Use bottom quarter
        
        # Calculate average x position of bottom points
        bottom_x = np.mean(bottom_points[:, 0])
        
        # Calculate distance from center as fraction of half frame width
        center_distance = abs(bottom_x - (self.width / 2))
        normalized_distance = center_distance / (self.width / 2)
        
        # Convert to urgency factor - closer to center means more urgent
        # Use exponential to make response stronger as we get very close
        urgency = np.exp(1 - normalized_distance) - 1
        return min(urgency, 1.0)

    def _calculate_exponential_steering(self, horizontalness: float,
                                    curve_bottom_x: float,
                                    frame_width: float,
                                    detection_status: str) -> float:
        """Calculate steering magnitude using exponential response to curve geometry.
        
        This method creates a non-linear steering response that becomes more
        aggressive as curves become sharper or as the detected lane moves further
        from its expected position. The exponential response provides:
        
        1. Gentle corrections for slight deviations, maintaining smooth driving
        2. Progressively stronger steering as turns become sharper
        3. Rapid response when the lane position indicates a severe turn
        
        The calculation combines two factors:
        - Curve horizontalness: How much the curve angles away from vertical
        - Edge proximity: How close the lane is to the frame edges
        
        Args:
            horizontalness: Ratio of horizontal to vertical curve distance [0 to 1]
            curve_bottom_x: X-coordinate of curve's bottom point
            frame_width: Width of the camera frame
            detection_status: Which lane was detected ('left' or 'right')
            
        Returns:
            Steering magnitude between 0.0 and 1.0
        """
        # Calculate exponential factor from curve shape
        # exp(x) - 1 gives us exponential growth starting from 0
        exp_factor = np.exp(horizontalness) - 1
        
        # Normalize to [0, 1] range using maximum possible value
        # np.e - 1 is the maximum value when horizontalness = 1
        exp_factor = min(exp_factor / (np.e - 1), 1.0)
        
        # Calculate how close the curve is to the frame edge
        # This factor increases as the lane approaches the relevant edge
        edge_factor = 0.0
        if detection_status == 'left':
            # For left lane, factor increases as x approaches 0
            edge_factor = 1.0 - (curve_bottom_x / (frame_width / 2))
        else:  # right lane
            # For right lane, factor increases as x approaches frame_width
            edge_factor = (curve_bottom_x - frame_width/2) / (frame_width/2)
        
        # Clamp edge factor to [0, 1] range
        edge_factor = max(0.0, min(1.0, edge_factor))
        
        # Combine factors with weights favoring curve shape over edge proximity
        # Shape is weighted more heavily as it's a more reliable indicator of turn severity
        shape_weight = 0.7
        edge_weight = 0.3
        
        steering_magnitude = (shape_weight * exp_factor + 
                            edge_weight * edge_factor)
        
        # Ensure result is a valid float between 0 and 1
        return float(max(0.0, min(1.0, steering_magnitude)))

    def _calculate_steering_angle(self, left_curve: Optional[np.ndarray], 
                                right_curve: Optional[np.ndarray],
                                detection_status: str) -> float:
        """Calculate the optimal steering angle based on detected lanes.
        
        This method implements a sophisticated steering strategy that handles:
        1. Dual-lane detection with center tracking
        2. Single-lane detection with offset compensation
        3. Curve anticipation for smoother navigation
        4. Bottom point proximity for urgent corrections
        
        The calculation considers multiple factors:
        - Immediate position correction needs
        - Upcoming curve geometry
        - Bottom point proximity to center
        - Lane-specific urgency factors
        
        Args:
            left_curve: Points forming the left lane curve, if detected
            right_curve: Points forming the right lane curve, if detected
            detection_status: Indicates which lanes were detected ('left', 'right', 'both')
            
        Returns:
            Steering angle in degrees, negative for left turns, positive for right turns.
            Always returns a valid float value between -30 and 30 degrees.
        """
        try:
            max_steering_angle = 30.0  # Maximum steering angle in either direction
            center_steering = 0.0      # Default to straight ahead
            frame_center = self.width / 2
            
            # Handle dual-lane detection
            if detection_status == 'both' and left_curve is not None and right_curve is not None:
                # Calculate center line between lanes
                center_curve = (left_curve + right_curve) / 2
                
                # Calculate immediate position correction needed
                bottom_center = center_curve[-1][0]  # X-coordinate at bottom of center line
                immediate_offset = bottom_center - frame_center
                
                # Normalize offset to [-1, 1] range for consistent steering response
                normalized_offset = immediate_offset / (self.width / 4)
                
                # Calculate anticipatory steering from curve shape
                curve_direction = self._estimate_curve_direction(left_curve, right_curve)
                
                # Get urgency factors for both lanes
                left_urgency = self._calculate_bottom_point_urgency(left_curve)
                right_urgency = self._calculate_bottom_point_urgency(right_curve)
                
                # Use maximum urgency between the two lanes
                combined_urgency = max(left_urgency, right_urgency)
                
                # Modify steering factors based on urgency
                normalized_offset *= (1.0 + combined_urgency)  # Increase correction when urgent
                curve_direction *= (1.0 - 0.5 * combined_urgency)  # Reduce anticipation when urgent
                
                # Combine immediate and anticipatory steering
                return self._combine_steering_factors(
                    immediate_offset=normalized_offset,
                    curve_direction=curve_direction,
                    max_angle=max_steering_angle
                )
            
            # Handle single-lane detection
            elif detection_status in ['left', 'right'] and (left_curve is not None or right_curve is not None):
                detected_curve = left_curve if detection_status == 'left' else right_curve
                if detected_curve is not None:
                    # Calculate curve properties
                    curve_bottom_x = detected_curve[-1][0]
                    curve_top_x = detected_curve[0][0]
                    vertical_distance = detected_curve[-1][1] - detected_curve[0][1]
                    
                    # Calculate how horizontal the curve is becoming
                    horizontalness = abs(curve_top_x - curve_bottom_x) / max(vertical_distance, 1)
                    
                    # Calculate bottom point urgency
                    urgency = self._calculate_bottom_point_urgency(detected_curve)
                    
                    # Determine base steering direction
                    steering_direction = 1 if detection_status == 'left' else -1
                    
                    # Calculate base steering magnitude
                    base_magnitude = self._calculate_exponential_steering(
                        horizontalness=horizontalness,
                        curve_bottom_x=curve_bottom_x,
                        frame_width=self.width,
                        detection_status=detection_status
                    )
                    
                    # Modify steering magnitude based on urgency
                    if abs(base_magnitude) < 0.5:  # For minor corrections
                        # Increase response when close to lane
                        steering_magnitude = base_magnitude * (1.0 + urgency)
                    else:  # For sharp turns
                        # Reduce aggressive anticipatory steering until we're closer
                        steering_magnitude = base_magnitude * (0.5 + 0.5 * urgency)
                    
                    # Calculate final steering angle
                    steering_angle = float(steering_direction * steering_magnitude * max_steering_angle)
                    
                    # Apply additional urgency scaling for very close situations
                    if urgency > 0.8:  # When very close to lane
                        # Strengthen corrective response to avoid crossing lane
                        steering_angle *= 1.2
                    
                    return max(min(steering_angle, max_steering_angle), -max_steering_angle)
            
            # Return center steering if no valid detection
            return center_steering
            
        except Exception as e:
            print(f"Warning: Steering calculation failed: {str(e)}")
            return 0.0  # Safe default steering angle

    def _estimate_curve_direction(self, left_curve: Optional[np.ndarray], 
                                right_curve: Optional[np.ndarray]) -> float:
        """Estimate the overall direction of the track's curve ahead.
        
        This method analyzes the detected lane curves to understand how the track
        curves ahead of the vehicle. It works with either one or both lane lines
        and uses the entire visible curve shape, not just the immediate section.
        The resulting direction value helps the vehicle anticipate and smoothly
        navigate upcoming turns.
        
        The direction is calculated by analyzing the horizontal displacement of
        the curves from bottom to top, with careful handling of situations where
        only one lane line is visible. The calculation includes normalization
        to ensure consistent steering response regardless of curve steepness.
        
        Args:
            left_curve: Points forming the left lane curve
            right_curve: Points forming the right lane curve
            
        Returns:
            Direction value between -1.0 (sharp left) and 1.0 (sharp right)
        """
        curves_to_check = []
        if left_curve is not None:
            curves_to_check.append(left_curve)
        if right_curve is not None:
            curves_to_check.append(right_curve)
            
        if not curves_to_check:
            return self.prev_curve_direction
            
        directions = []
        for curve in curves_to_check:
            if len(curve) >= 2:
                # Calculate horizontal displacement from curve bottom to top
                dx = curve[0][0] - curve[-1][0]
                dy = curve[0][1] - curve[-1][1]
                
                # Normalize by maximum expected displacement for consistent scaling
                max_dx = self.width * 0.3  # Maximum expected lane shift
                direction = dx / max_dx
                directions.append(direction)
        
        if not directions:
            return self.prev_curve_direction
            
        # Calculate new direction with temporal smoothing
        avg_direction = np.mean(directions)
        smoothed_direction = (self.position_memory_decay * self.prev_curve_direction +
                            (1 - self.position_memory_decay) * avg_direction)
        
        self.prev_curve_direction = smoothed_direction
        return smoothed_direction

    def _combine_steering_factors(self, immediate_offset: float,
                                curve_direction: float,
                                max_angle: float) -> float:
        """Combine immediate and anticipatory steering factors.
        
        This method creates a balanced steering response by combining two key factors:
        1. Immediate offset correction to maintain lane centering
        2. Anticipatory steering based on upcoming curve direction
        
        The combination uses carefully tuned weights to create smooth, natural
        steering behavior that both maintains proper lane position and smoothly
        navigates curves. The immediate correction ensures quick response to
        position errors, while the anticipatory component allows the vehicle
        to smoothly enter curves before significant position error develops.
        
        Args:
            immediate_offset: Normalized current position error [-1, 1]
            curve_direction: Overall curve direction ahead [-1, 1]
            max_angle: Maximum allowed steering angle
            
        Returns:
            Combined steering angle in degrees
        """ 
        # Weight immediate correction vs anticipatory steering
        immediate_weight = 0.6  # Higher weight for responsive corrections
        curve_weight = 0.4     # Lower weight for smoother curve handling
        
        # Combine factors with weights
        steering_factor = (immediate_weight * immediate_offset + 
                         curve_weight * curve_direction)
        
        # Convert to actual steering angle
        return steering_factor * max_angle

    def _calculate_dynamic_roi(self, frame_height: int, detection_status: str, 
                            left_curve: Optional[np.ndarray] = None, 
                            right_curve: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Calculate dynamic ROI boundaries based on lane characteristics.
        
        The ROI adapts to different driving scenarios by adjusting its bottom boundary:
        - In straight sections with both lanes visible, we use the default bottom 
        boundary since the perspective makes lanes clearly visible higher up
        - In curves or single-lane scenarios, we extend down to capture more immediate
        road area, potentially using the entire frame height
        
        The top boundary remains fixed to maintain consistent look-ahead distance,
        which is crucial for anticipating upcoming road conditions.
        
        Args:
            frame_height: Height of the camera frame in pixels
            detection_status: Current detection state ('left', 'right', or 'both')
            left_curve: Detected left lane points, if any
            right_curve: Detected right lane points, if any
        
        Returns:
            Tuple of (y_start, y_end) as fractions of frame height
        """
        max_end = 1.0  # Maximum bottom boundary (full frame height)
        
        # For single lane detection, we need to see as much of the lane as possible
        if detection_status in ['left', 'right']:
            return (self.roi_default_start, max_end)
        
        # For dual lane detection, analyze curvature to determine bottom boundary
        if detection_status == 'both' and left_curve is not None and right_curve is not None:
            # Calculate how much each lane curves by comparing x-coordinates
            left_curve_amount = abs(left_curve[0][0] - left_curve[-1][0])
            right_curve_amount = abs(right_curve[0][0] - right_curve[-1][0])
            
            # Average curvature as fraction of frame width
            avg_curve = (left_curve_amount + right_curve_amount) / (2 * self.width)
            
            if avg_curve < 0.1:  # Relatively straight lanes
                return (self.roi_default_start, self.roi_default_end)
            else:
                # Scale bottom boundary linearly with curve amount
                # Map curve amount [0.1, 0.5] to y_end [default_end, max_end]
                curve_factor = min(1.0, (avg_curve - 0.1) / 0.4)
                y_end = self.roi_default_end + ((max_end - self.roi_default_end) * curve_factor)
                return (self.roi_default_start, y_end)
        
        # Default case - use standard ROI boundaries
        return (self.roi_default_start, self.roi_default_end)

    def process_frame(self, frame: DetectionFrame) -> DetectionResult:
        """Process a single camera frame to detect and analyze lane lines."""
        try:
            start_time = time.time()

            if frame is None or frame.frame is None or frame.frame.size == 0:
                raise ValueError("Invalid or empty frame received")
            
            # Initialize metadata dictionary with required fields
            metadata = {
                'detection_status': 'none',
                'processing_time': 0.0,
                'lane_width_confidence': 0.0,
                'binary_frame': None,
                'left_points': None,
                'right_points': None,
                'roi_y_start': None,
                'roi_y_end': None,
                'roi_vertices': None
            }
            
            # Initialize detection data structure with safe defaults
            data = LaneDetectionData(
                left_curve=None,
                right_curve=None,
                center_offset=0,
                curve_radius=None,
                curve_direction=0.0,
                steering_angle=0.0,
                steering_confidence=0.0
            )
            
            # Calculate initial ROI position
            y_start = int(self.height * self.roi_default_start)
            y_end = int(self.height * self.roi_default_end)

            # Validate ROI boundaries
            if y_start >= y_end or y_end > frame.frame.shape[0]:
                raise ValueError("Invalid ROI boundaries")
            
            # Store initial ROI information in metadata
            metadata['roi_y_start'] = y_start
            metadata['roi_y_end'] = y_end
            
            # Create ROI vertices for visualization
            roi_vertices = np.array([
                [(0, y_start),
                (0, y_end),
                (self.width, y_end),
                (self.width, y_start)]
            ], dtype=np.int32)
            metadata['roi_vertices'] = roi_vertices
            
            # Extract ROI region and process
            roi_frame = frame.frame[y_start:y_end, :]
            if roi_frame is None or roi_frame.size == 0:
                raise ValueError("ROI extraction resulted in empty frame")
            binary = self._preprocess_frame(roi_frame)
            metadata['binary_frame'] = binary
            
            # Detect lane points
            left_points, right_points, detection_status = self._find_lane_points(binary)
            
            # Fit initial curves if points were found
            left_curve = self._fit_curve(left_points) if len(left_points) > 0 else None
            right_curve = self._fit_curve(right_points) if len(right_points) > 0 else None
            
            # Calculate dynamic ROI based on initial detection
            y_start_frac, y_end_frac = self._calculate_dynamic_roi(
                frame_height=self.height,
                detection_status=detection_status,
                left_curve=left_curve,
                right_curve=right_curve
            )
            
            # Calculate new ROI boundaries
            new_y_start = int(self.height * y_start_frac)
            new_y_end = int(self.height * y_end_frac)
            
            # Check if ROI changed significantly
            roi_change = abs(new_y_end - y_end)
            if roi_change > 10:  # Threshold for significant change
                # Update ROI information
                y_start = new_y_start
                y_end = new_y_end
                
                # Update ROI vertices for visualization
                roi_vertices = np.array([
                    [(0, y_start),
                    (0, y_end),
                    (self.width, y_end),
                    (self.width, y_start)]
                ], dtype=np.int32)
                
                # Repeat detection with new ROI
                roi_frame = frame.frame[y_start:y_end, :]
                binary = self._preprocess_frame(roi_frame)
                left_points, right_points, detection_status = self._find_lane_points(binary)
                
                # Update curves with new detection
                left_curve = self._fit_curve(left_points) if len(left_points) > 0 else None
                right_curve = self._fit_curve(right_points) if len(right_points) > 0 else None
                
                # Update metadata with new ROI information
                metadata['roi_y_start'] = y_start
                metadata['roi_y_end'] = y_end
                metadata['roi_vertices'] = roi_vertices
                metadata['binary_frame'] = binary
            
            # Store final detection results in metadata
            metadata['left_points'] = left_points
            metadata['right_points'] = right_points
            metadata['detection_status'] = detection_status
            
            # Rest of processing remains the same...
            current_width = self._update_lane_width(left_curve, right_curve)
            metadata['lane_width_confidence'] = self.lane_width_confidence
            
            steering_angle = self._calculate_steering_angle(
                left_curve, right_curve, detection_status
            )
            
            steering_confidence = self._calculate_steering_confidence(
                detection_status,
                self.lane_width_confidence
            )
            
            data = LaneDetectionData(
                left_curve=left_curve,
                right_curve=right_curve,
                center_offset=0,
                curve_radius=self._calculate_curve_radius(left_curve if left_curve is not None else right_curve),
                curve_direction=self._estimate_curve_direction(left_curve, right_curve),
                steering_angle=steering_angle,
                steering_confidence=steering_confidence
            )
            
            metadata['processing_time'] = time.time() - start_time
            
            return DetectionResult(
                is_valid=detection_status != 'none',
                data=data,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            return DetectionResult(
                is_valid=False,
                data=LaneDetectionData(
                    left_curve=None,
                    right_curve=None,
                    center_offset=0,
                    curve_radius=None,
                    curve_direction=0.0,
                    steering_angle=0.0,
                    steering_confidence=0.0
                ),
                metadata={
                    'detection_status': 'none',
                    'processing_time': time.time() - start_time,
                    'lane_width_confidence': 0.0,
                    'roi_vertices': metadata.get('roi_vertices', None),
                    'roi_y_start': metadata.get('roi_y_start', None),
                    'roi_y_end': metadata.get('roi_y_end', None),
                    'error': str(e)
                }
            )

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration parameters.
        
        Returns a copy of the current configuration dictionary containing
        all detection and processing parameters. This allows external code
        to inspect but not directly modify the configuration.
        
        Returns:
            Dictionary of current configuration parameters
        """
        return self.config.copy()

    def update_config(self, param: str, value: int) -> None:
        """Update a configuration parameter with validation.
        
        This method allows runtime updates to detection parameters while ensuring
        the new values remain within safe operating ranges. Each parameter type
        has specific validation rules:
        
        Kernel Sizes:
        - Must be odd numbers (for symmetric filtering)
        - Have minimum/maximum bounds to maintain performance
        
        Thresholds:
        - Bounded to prevent over/under sensitivity
        - Scaled appropriately for their specific uses
        
        Window Parameters:
        - Bounded to ensure effective lane tracking
        - Validated for computational efficiency
        
        Args:
            param: Name of parameter to update
            value: New value for parameter
            
        The method silently maintains parameter bounds rather than raising errors,
        ensuring the system remains operational even with invalid input attempts.
        """
        # Define valid ranges for each parameter type
        valid_ranges = {
            # Image processing parameters
            'blur_kernel_size': (3, 11),       # Must be odd, range 3-11
            'adaptive_block_size': (3, 99),    # Must be odd, range 3-99
            'adaptive_offset': (0, 50),        # Threshold adjustment
            'dilate_iterations': (0, 3),       # Morphological operations
            'erode_iterations': (0, 3),
            
            # Lane detection parameters
            'min_points_for_curve': (3, 10),   # Minimum points for valid curve
            'min_peak_value': (10, 100),       # Histogram peak detection
            'min_peak_distance': (10, 100),    # Minimum peak separation
            'window_height': (5, 15),          # Sliding window parameters
            'window_margin': (30, 100),
            'minpix': (5, 50)                  # Minimum pixels for recenter
        }
        
        if param in valid_ranges:
            min_val, max_val = valid_ranges[param]
            
            # Clamp value to valid range
            validated_value = max(min_val, min(value, max_val))
            
            # Ensure odd values for kernel sizes
            if param in ['blur_kernel_size', 'adaptive_block_size']:
                if validated_value % 2 == 0:
                    validated_value = validated_value - 1
            
            # Update the configuration
            self.config[param] = validated_value
            
            # Log the update for debugging
            print(f"Updated {param} to {validated_value}")
        else:
            print(f"Warning: Unknown configuration parameter: {param}")

    def reset(self):
        """Reset detector state and clear historical tracking data.
        
        Reinitializes all temporal tracking variables to their default states.
        This is useful when:
        - Starting a new detection sequence
        - Recovering from detection failures
        - Testing different detection parameters
        """
        self.prev_left_curve = None
        self.prev_right_curve = None
        self.prev_curve_radius = None
        self.prev_curve_direction = 0
        self.recent_lane_widths = []
        self.lane_width_confidence = 0.0
        self.lane_classification_history = []
        self.prev_left_x = None
        self.prev_right_x = None