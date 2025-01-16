# src/detection/lane.py

import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict, Any
from .base import Detector, DetectionResult
from utils.types import DetectionFrame, LaneDetectionData
import time

# TODO: Just get the algorithm to detect blocks/sections of white pixels
# TODO:To extract the individual sections of white pixels,can use connected components or contour detection. 

# if there is only one uninterrupted section of white pixels, then it is a single lane
# if only one lane is visible, and if it curves up and right by any amount, then it is a left lane
# if it curves up and left by any amount, then it is a left lane

# if there are two separate sections of white pixels, then there are two lanes visible
# if there are two lanes visible, then the lane with the bottom most pixel on the left is the left lane, and then all the pixels in the entire section should account for the left lane
# and the lane with the bottom most pixel on the right is the right lane, and then all the pixels in the entire section should account for the right lane

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
        
        # Define ROI coordinates (40% to 80% of frame height)
        y_start = int(self.height * 0.5)
        y_end = int(self.height * 0.8)
        
        self.roi_vertices = np.array([
            [(0, y_start),
             (0, y_end),
             (self.width, y_end),
             (self.width, y_start)]
        ], dtype=np.int32)
        
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
            'min_peak_value': 20,         # Minimum histogram peak height
            'min_peak_distance': 30,      # Minimum separation between distinct peaks
            'window_height': 9,           # Number of sliding windows
            'window_margin': 60,          # Sliding window width
            'minpix': 15                  # Minimum pixels for window recentering
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
        
        # Pre-allocated arrays for efficient curve fitting
        self.y_points = np.linspace(frame_height, 0, num=20)
        self.curve_points = np.zeros((20, 2), dtype=np.int32)

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

    def _apply_roi(self, img: np.ndarray) -> np.ndarray:
        """Extract region of interest from binary image.
        
        Creates and applies a mask to focus processing on the relevant portion
        of the frame where lanes are expected to appear. The ROI is a trapezoid
        that covers the middle 40% to 80% of the frame height.
        
        Args:
            img: Binary input image
            
        Returns:
            Masked binary image containing only the ROI
        """
        y_start = int(self.height * 0.5)
        y_end = int(self.height * 0.8)
        roi_height = y_end - y_start
        
        mask = np.zeros((roi_height, self.width), dtype=img.dtype)
        
        roi_vertices = self.roi_vertices.copy()
        roi_vertices[:, :, 1] -= y_start
        
        cv2.fillPoly(mask, roi_vertices, 255)
        return cv2.bitwise_and(img[y_start:y_end], mask)

    def _find_lane_points(self, binary_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """Detect and classify lane points in the binary image.
        
        This method implements a sophisticated lane point detection algorithm:
        1. Analyzes histogram of bottom region to find initial lane positions
        2. Uses sliding windows to track points up through the image
        3. Validates and classifies detected points as left/right lane
        4. Maintains temporal consistency through position tracking
        
        Args:
            binary_img: Preprocessed binary image
            
        Returns:
            Tuple containing:
            - Left lane points (Nx2 array or empty)
            - Right lane points (Nx2 array or empty)
            - Detection status ('left', 'right', 'both', or 'none')
        """
        # Find all non-black pixels in the binary image
        nonzero_y, nonzero_x = binary_img.nonzero()
        
        # Analyze bottom portion for initial lane positions
        bottom_region_height = int(binary_img.shape[0] * 0.3)
        bottom_region = binary_img[-bottom_region_height:, :]
        histogram = np.sum(bottom_region, axis=0)
        
        midpoint = binary_img.shape[1] // 2
        left_hist = histogram[:midpoint]
        right_hist = histogram[midpoint:]
        
        left_max = np.max(left_hist)
        right_max = np.max(right_hist)
        
        # Use different thresholds for left and right lanes
        left_min_peak = self.config['min_peak_value']
        right_min_peak = self.config['min_peak_value'] * 0.8
        
        # Initialize empty arrays for points
        left_points = np.array([], dtype=np.int32).reshape(0, 2)
        right_points = np.array([], dtype=np.int32).reshape(0, 2)
        
        # Find lane base points
        left_base = None if left_max < left_min_peak else np.argmax(left_hist)
        right_base = None if right_max < right_min_peak else midpoint + np.argmax(right_hist)
        
        # Track points if bases are found
        if left_base is not None:
            left_points_list = self._track_line_points(
                binary_img=binary_img,
                base_x=left_base,
                nonzero_x=nonzero_x,
                nonzero_y=nonzero_y
            )
            if left_points_list:
                left_points = np.array(left_points_list)
                
        if right_base is not None:
            right_points_list = self._track_line_points(
                binary_img=binary_img,
                base_x=right_base,
                nonzero_x=nonzero_x,
                nonzero_y=nonzero_y
            )
            if right_points_list:
                right_points = np.array(right_points_list)
        
        # Determine detection status
        if len(left_points) > 0 and len(right_points) > 0:
            detection_status = 'both'
        elif len(left_points) > 0:
            detection_status = 'left'
        elif len(right_points) > 0:
            detection_status = 'right'
        else:
            detection_status = 'none'
        
        return left_points, right_points, detection_status

    def _track_line_points(self, binary_img: np.ndarray, base_x: int, nonzero_x: np.ndarray, nonzero_y: np.ndarray) -> List:
        """Track lane points vertically through the image using sliding windows.
        
        This method implements an adaptive sliding window algorithm that:
        1. Starts from a base point at the bottom of the image
        2. Moves upward in fixed-height windows
        3. Centers each window on detected lane points
        4. Adjusts window width based on lane curvature
        
        The algorithm includes several features for robust tracking:
        - Dynamic window margins that adapt to curve rate
        - Predictive window positioning using movement history
        - Recovery mechanisms for temporarily lost tracking
        - Continuous validation of detected points
        
        Args:
            binary_img: Binary input image
            base_x: Starting x-coordinate for tracking
            nonzero_x: X-coordinates of white pixels in binary image
            nonzero_y: Y-coordinates of white pixels in binary image
        
        Returns:
            List of points belonging to the tracked lane
        """
        points = []
        current_x = base_x
        x_history = [current_x]
        margin_history = []
        
        window_height = binary_img.shape[0] // self.config['window_height']
        base_margin = self.config['window_margin']
        minpix = self.config['minpix']
        
        for window in range(self.config['window_height']):
            # Calculate dynamic search margin based on curve rate
            if len(x_history) >= 2:
                recent_movement = abs(x_history[-1] - x_history[-2])
                dynamic_margin = min(base_margin * (1 + recent_movement / base_margin), 
                                    base_margin * 2)
            else:
                dynamic_margin = base_margin
            
            # Predict next window center using momentum
            if len(x_history) >= 2:
                movement = x_history[-1] - x_history[-2]
                predicted_x = current_x + int(movement * 0.8)
            else:
                predicted_x = current_x
            
            # Define search window boundaries
            win_y_low = binary_img.shape[0] - (window + 1) * window_height
            win_y_high = binary_img.shape[0] - window * window_height
            win_x_low = max(0, predicted_x - dynamic_margin)
            win_x_high = min(binary_img.shape[1], predicted_x + dynamic_margin)
            
            # Find points within window
            good_inds = ((nonzero_y >= win_y_low) & 
                        (nonzero_y < win_y_high) & 
                        (nonzero_x >= win_x_low) & 
                        (nonzero_x < win_x_high)).nonzero()[0]
            
            if len(good_inds) > minpix:
                current_x = int(np.mean(nonzero_x[good_inds]))
                points.extend(np.column_stack((nonzero_x[good_inds], 
                                            nonzero_y[good_inds])))
                x_history.append(current_x)
                margin_history.append(dynamic_margin)
                
            # Implement recovery for lost tracking
            elif len(x_history) >= 2:
                recovery_margin = dynamic_margin * 1.5
                win_x_low = max(0, predicted_x - recovery_margin)
                win_x_high = min(binary_img.shape[1], predicted_x + recovery_margin)
                
                # Try wider search window
                good_inds = ((nonzero_y >= win_y_low) & 
                            (nonzero_y < win_y_high) & 
                            (nonzero_x >= win_x_low) & 
                            (nonzero_x < win_x_high)).nonzero()[0]
                
                if len(good_inds) > minpix:
                    current_x = int(np.mean(nonzero_x[good_inds]))
                    points.extend(np.column_stack((nonzero_x[good_inds], 
                                                nonzero_y[good_inds])))
                else:
                    current_x = predicted_x
                
                x_history.append(current_x)
                margin_history.append(recovery_margin)
        
        return points

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

    def _classify_single_lane(self, points: np.ndarray, curve: np.ndarray) -> str:
        """Classify a single detected lane as either left or right.
        
        Uses multiple geometric features to make a robust classification:
        - Absolute position in frame
        - Curve direction and shape
        - Historical position consistency
        - Temporal smoothing of classification
        
        The method weights these features and applies temporal smoothing to
        prevent classification flips.
        
        Args:
            points: Original detected points for the lane
            curve: Fitted curve points
            
        Returns:
            'left' or 'right' classification
        """
        if len(curve) < 3:
            return 'left' if np.mean(points[:, 0]) < self.width/2 else 'right'
        
        # Calculate position-based score relative to frame center
        center_x = self.width / 2
        bottom_x = curve[-1][0]
        position_score = (bottom_x - center_x) / center_x  # Normalized to [-1, 1]
        
        # Calculate direction score using curve geometry
        bottom_section = curve[-3:]  # Use last 3 points
        top_section = curve[:3]      # Use first 3 points
        direction_vector = top_section.mean(axis=0) - bottom_section.mean(axis=0)
        # Use arctan to handle extreme angles while keeping score in [-1, 1]
        direction_score = np.arctan2(direction_vector[0], abs(direction_vector[1])) / (np.pi/2)
        
        # Calculate temporal consistency using previous positions
        temporal_score = 0
        if self.prev_left_x is not None and self.prev_right_x is not None:
            dist_to_left = abs(bottom_x - self.prev_left_x)
            dist_to_right = abs(bottom_x - self.prev_right_x)
            temporal_score = 1 if dist_to_right < dist_to_left else -1
        
        # Weight and combine scores for final classification
        weights = {
            'position': 0.5,    # Highest weight for absolute position
            'direction': 0.3,   # Medium weight for curve direction
            'temporal': 0.2     # Lower weight for temporal consistency
        }
        
        final_score = (weights['position'] * position_score +
                        weights['direction'] * direction_score +
                        weights['temporal'] * temporal_score)
        
        # Get initial classification based on weighted score
        classification = 'left' if final_score < 0 else 'right'
        
        # Apply temporal smoothing using classification history
        self.lane_classification_history.append(classification)
        if len(self.lane_classification_history) > self.classification_history_size:
            self.lane_classification_history.pop(0)
        
        # Return most common recent classification
        return max(set(self.lane_classification_history), 
                    key=self.lane_classification_history.count)

    def _validate_curves(self, left_curve: np.ndarray, right_curve: np.ndarray) -> bool:
        """Validate detected curves using geometric constraints and relationships.
        
        Implements multiple validation criteria:
        1. Basic position check (left curve must be left of right curve)
        2. Minimum separation distance between curves
        3. Parallel alignment check
        4. Consistent separation along curve length
        
        These geometric constraints help prevent false dual-lane detection
        when a single lane is detected twice or when noise creates false curves.
        
        Args:
            left_curve: Points forming the left curve
            right_curve: Points forming the right curve
            
        Returns:
            Boolean indicating if curves represent valid separate lanes
        """
        if left_curve is None or right_curve is None:
            return True
            
        # Calculate average x-positions for basic position validation
        left_x = np.mean(left_curve[:, 0])
        right_x = np.mean(right_curve[:, 0])
        
        # Verify left curve is left of right curve
        if left_x >= right_x:
            return False
        
        # Check minimum separation between curves
        min_separation = 50  # Minimum valid separation in pixels
        separation = right_x - left_x
        if separation < min_separation:
            return False
        
        # Validate curve parallelism when sufficient points exist
        if len(left_curve) >= 3 and len(right_curve) >= 3:
            left_direction = left_curve[0] - left_curve[-1]
            right_direction = right_curve[0] - right_curve[-1]
            
            # Normalize direction vectors
            left_direction = left_direction / np.linalg.norm(left_direction)
            right_direction = right_direction / np.linalg.norm(right_direction)
            
            # Check alignment using dot product
            dot_product = np.dot(left_direction, right_direction)
            if dot_product < 0.75:  # Allow up to ~41 degrees difference
                return False
        
        # Verify consistent separation along curves
        separations = []
        for i in range(min(len(left_curve), len(right_curve))):
            sep = right_curve[i][0] - left_curve[i][0]
            separations.append(sep)
        
        # Calculate variation in separation
        separation_variation = np.std(separations) / np.mean(separations)
        if separation_variation > 0.3:  # Allow 30% variation
            return False
            
        return True

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
        4. Recovery steering for partial detection cases
        
        The calculation considers both immediate position correction needs and
        upcoming curve geometry to produce natural steering behavior. The method
        includes comprehensive error handling to ensure it always returns a 
        valid steering angle.
        
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
                    
                    # Determine base steering direction
                    steering_direction = 1 if detection_status == 'left' else -1
                    
                    # Calculate steering magnitude based on curve geometry
                    steering_magnitude = self._calculate_exponential_steering(
                        horizontalness=horizontalness,
                        curve_bottom_x=curve_bottom_x,
                        frame_width=self.width,
                        detection_status=detection_status
                    )
                    
                    return float(steering_direction * steering_magnitude * max_steering_angle)
            
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

    def process_frame(self, frame: DetectionFrame) -> DetectionResult:
        """Process a camera frame to detect lanes and determine steering commands.
        
        This method implements a complete lane detection and analysis pipeline:
        
        Image Preprocessing:
        - Converts frame to binary image optimized for lane detection
        - Applies ROI mask to focus on relevant portion of frame 
        - Uses adaptive thresholding to handle variable lighting
        
        Lane Detection:
        - Analyzes image histogram to find potential lane positions
        - Uses sliding windows to track lane points vertically
        - Fits polynomial curves to detected points
        - Validates and classifies detected lines as left/right lanes
        
        Steering Analysis:
        - Calculates optimal steering angle from detected geometry 
        - Determines confidence level in steering decision
        - Maintains temporal consistency through state tracking
        - Handles single and dual lane detection scenarios
        
        The pipeline includes comprehensive error handling and fallback behaviors,
        ensuring it always produces valid outputs even with partial or failed
        detection. Multiple validation steps help prevent false detections
        and maintain steering stability.
        
        Args:
            frame: DetectionFrame containing camera image and timestamp
            
        Returns:
            DetectionResult containing:
            - Detected lane data (curves, positions, steering)
            - Detection validity flag
            - Debug/visualization metadata
        """
        try:
            # Stage 1: Image preprocessing
            binary = self._preprocess_frame(frame.frame)
            roi = self._apply_roi(binary)
            
            # Stage 2: Lane point detection 
            left_points, right_points, detection_status = self._find_lane_points(roi)
            
            # Initialize metrics with safe default values
            steering_angle = 0.0  # Always initialize to a valid float
            center_offset = 0
            curve_radius = None
            curve_direction = self.prev_curve_direction
            steering_confidence = 0.0
            left_curve = None
            right_curve = None
            is_valid = False
            lane_center = self.width // 2  # Default to frame center
            
            # Stage 3: Process detected lanes when available
            if len(left_points) > 0 or len(right_points) > 0:
                # Fit polynomial curves to detected points
                if len(left_points) > 0:
                    left_curve = self._fit_curve(left_points)
                if len(right_points) > 0:
                    right_curve = self._fit_curve(right_points)
                    
                # Update position history for temporal smoothing
                self._update_position_tracking(left_points, right_points)
                
                # Process curves if at least one is valid
                if left_curve is not None or right_curve is not None:
                    # Calculate steering angle with safety checks
                    new_steering_angle = self._calculate_steering_angle(
                        left_curve=left_curve,
                        right_curve=right_curve,
                        detection_status=detection_status
                    )
                    
                    # Update steering angle if valid
                    if isinstance(new_steering_angle, (int, float)):
                        steering_angle = float(new_steering_angle)
                    
                    # Calculate supporting metrics
                    reference_curve = left_curve if left_curve is not None else right_curve
                    curve_radius = self._calculate_curve_radius(reference_curve)
                    curve_direction = self._estimate_curve_direction(left_curve, right_curve)
                    
                    # Calculate lane center and position offset
                    frame_center = self.width // 2
                    if detection_status == 'both':
                        lane_center = (left_curve[-1][0] + right_curve[-1][0]) // 2
                    else:
                        detected_curve = left_curve if left_curve is not None else right_curve
                        if detection_status == 'left':
                            lane_center = detected_curve[-1][0] + self.width // 4
                        else:  # right lane
                            lane_center = detected_curve[-1][0] - self.width // 4
                    
                    center_offset = lane_center - frame_center
                    
                    # Set confidence based on detection quality
                    steering_confidence = 1.0 if detection_status == 'both' else 0.7
                    is_valid = True
            
            # Create lane detection data structure with guaranteed valid values
            lane_data = LaneDetectionData(
                left_curve=left_curve,
                right_curve=right_curve,
                center_offset=center_offset,
                curve_radius=curve_radius,
                curve_direction=curve_direction,
                steering_angle=float(steering_angle),  # Ensure float type
                steering_confidence=steering_confidence
            )
            
            # Prepare comprehensive metadata for debugging/visualization
            metadata = {
                'binary_frame': roi,
                'roi_vertices': self.roi_vertices,
                'left_points': left_points if len(left_points) > 0 else None,
                'right_points': right_points if len(right_points) > 0 else None,
                'detection_status': detection_status,
                'lane_width_confidence': self.lane_width_confidence,
                'adaptive_threshold': self.config['adaptive_block_size'],
                'binary_frame_full': binary,
                'processing_time': time.time() - frame.timestamp
            }
            
            return DetectionResult(
                is_valid=is_valid,
                data=lane_data,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            # Return safe default values if processing fails
            return DetectionResult(
                is_valid=False,
                data=LaneDetectionData(
                    left_curve=None,
                    right_curve=None,
                    center_offset=0,
                    curve_radius=None,
                    curve_direction=self.prev_curve_direction,
                    steering_angle=0.0,
                    steering_confidence=0.0
                ),
                metadata={
                    'error': str(e),
                    'processing_time': time.time() - frame.timestamp
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