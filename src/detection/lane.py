# src/detection/lane.py

import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict, Any
from .base import Detector, DetectionResult
from utils.types import DetectionFrame, LaneDetectionData

class LaneDetector(Detector):
    """Enhanced lane detection system for limited FOV and single-lane scenarios.
    
    This detector is specifically designed to work with:
    1. A camera with limited field of view where lanes may start partway up the frame
    2. Situations where only one lane line is visible
    3. Sharp curves where lanes may partially leave the frame
    4. Variable lighting conditions through adaptive thresholding
    """
    
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        self.width = frame_width
        self.height = frame_height
        
        # Define ROI for middle 60% of frame
        y_start = int(self.height * 0.2)  # Start at 20% from top
        y_end = int(self.height * 0.8)    # End at 80% from top
        
        self.roi_vertices = np.array([
            [(0, y_start),              # Top left
             (0, y_end),                # Bottom left
             (self.width, y_end),       # Bottom right
             (self.width, y_start)]     # Top right
        ], dtype=np.int32)
        
        # Configuration parameters
        self.config = {
            'blur_kernel_size': 7,        # Kernel size for noise reduction
            'adaptive_block_size': 45,    # Block size for adaptive threshold
            'adaptive_offset': 25,        # Constant subtracted from mean
            'dilate_iterations': 1,       # Number of dilation steps
            'erode_iterations': 1,        # Number of erosion steps
            'min_points_for_curve': 4,    # Minimum points needed for curve fitting
            'curve_smoothing_factor': 0.9 
        }
        
        # State tracking
        self.prev_left_curve = None
        self.prev_right_curve = None
        self.prev_curve_radius = None
        self.prev_curve_direction = 0
        
        # Track recent observations for lane width learning
        self.recent_lane_widths = []
        self.max_width_history = 10
        self.lane_width_confidence = 0.0
        
        # Pre-allocate arrays for curve fitting
        self.y_points = np.linspace(frame_height, 0, num=20)
        self.curve_points = np.zeros((20, 2), dtype=np.int32)

    def update_config(self, param: str, value: int) -> None:
        """Update a configuration parameter with validation."""
        valid_ranges = {
            'blur_kernel_size': (3, 11),
            'adaptive_block_size': (3, 99),
            'adaptive_offset': (0, 50),
            'dilate_iterations': (0, 3),
            'erode_iterations': (0, 3),
            'min_points_for_curve': (3, 10)
        }
        
        if param in valid_ranges:
            min_val, max_val = valid_ranges[param]
            validated_value = max(min_val, min(value, max_val))
            
            # Ensure odd values for kernel sizes
            if param in ['blur_kernel_size', 'adaptive_block_size']:
                validated_value = validated_value if validated_value % 2 == 1 else validated_value - 1
                
            self.config[param] = validated_value

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Prepare frame using adaptive thresholding for better lane isolation."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        k_size = self.config['blur_kernel_size']
        blurred = cv2.GaussianBlur(gray, (k_size, k_size), 0)
        
        # Use adaptive thresholding to handle variable lighting
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.config['adaptive_block_size'],
            self.config['adaptive_offset']
        )
        
        # Clean up binary image
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.dilate(binary, kernel, 
                           iterations=self.config['dilate_iterations'])
        binary = cv2.erode(binary, kernel, 
                           iterations=self.config['erode_iterations'])
        
        return binary

    def _apply_roi(self, img: np.ndarray) -> np.ndarray:
        """Apply region of interest mask for middle portion of frame."""
        # Calculate ROI boundaries
        y_start = int(self.height * 0.2)  # Start at 20% from top
        y_end = int(self.height * 0.8)    # End at 80% from top
        roi_height = y_end - y_start
        
        # Create mask for ROI area
        mask = np.zeros((roi_height, self.width), dtype=img.dtype)
        
        # Adjust vertices for the mask
        roi_vertices = self.roi_vertices.copy()
        roi_vertices[:, :, 1] -= y_start  # Shift vertices to start from 0 in the mask
        
        # Fill only the necessary part
        cv2.fillPoly(mask, roi_vertices, 255)
        
        # Apply mask only to ROI area
        return cv2.bitwise_and(img[y_start:y_end], mask)

    def _update_lane_width(self, left_curve: Optional[np.ndarray], 
                          right_curve: Optional[np.ndarray]) -> float:
        """Track and learn lane width from observations."""
        if left_curve is not None and right_curve is not None:
            # Measure width at several points and take the average
            widths = []
            for i in range(min(len(left_curve), len(right_curve))):
                width = abs(right_curve[i][0] - left_curve[i][0])
                widths.append(width)
            
            if not widths:  # If no widths were measured
                return None
            
            current_width = np.mean(widths)
            
            # Update recent measurements
            self.recent_lane_widths.append(current_width)
            if len(self.recent_lane_widths) > self.max_width_history:
                self.recent_lane_widths.pop(0)
            
            # Update confidence based on consistency
            if len(self.recent_lane_widths) >= 3:
                std_dev = np.std(self.recent_lane_widths)
                mean_width = np.mean(self.recent_lane_widths)
                
                # Avoid division by zero and handle edge cases
                if mean_width > 0:
                    # Normalize std_dev by mean_width to get relative variation
                    relative_variation = std_dev / mean_width
                    # Convert to confidence score (0 to 1)
                    self.lane_width_confidence = max(0, min(1, 1 - relative_variation))
                else:
                    self.lane_width_confidence = 0.0
            
            return current_width
        
        elif len(self.recent_lane_widths) > 0:
            return np.mean(self.recent_lane_widths)
        
        return None

    def _find_lane_points(self, binary_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """Basic lane point detection using sliding windows."""
        # Get ROI offset for coordinate adjustment
        y_start = int(self.height * 0.2)
        
        # Find all white pixels
        nonzero_y, nonzero_x = binary_img.nonzero()
        
        # Take a histogram of the bottom half of the ROI image
        histogram = np.sum(binary_img[binary_img.shape[0]//2:, :], axis=0)
        
        # Find the peak of the left and right halves of the histogram
        midpoint = binary_img.shape[1]//2
        
        # Find peaks with lower threshold
        min_peak_value = 20  # Lower minimum peak height
        
        # Find left base
        left_hist = histogram[:midpoint]
        left_base = np.argmax(left_hist) if np.max(left_hist) > min_peak_value else None
        
        # Find right base
        right_hist = histogram[midpoint:]
        right_base = midpoint + np.argmax(right_hist) if np.max(right_hist) > min_peak_value else None
        
        # Set height of windows
        window_height = binary_img.shape[0]//9
        n_windows = 9
        
        # Set the width of the windows +/- margin
        margin = 80  # Increased margin
        
        # Set minimum number of pixels to recenter window
        minpix = 15  # Lower minimum pixels
        
        # Lists to store lane indices
        left_points = []
        right_points = []
        
        # Process left lane if base found
        if left_base is not None:
            leftx_current = left_base
            for window in range(n_windows):
                win_y_low = binary_img.shape[0] - (window + 1) * window_height
                win_y_high = binary_img.shape[0] - window * window_height
                
                win_xleft_low = max(0, leftx_current - margin)
                win_xleft_high = min(binary_img.shape[1], leftx_current + margin)
                
                good_left_inds = ((nonzero_y >= win_y_low) & 
                                 (nonzero_y < win_y_high) & 
                                 (nonzero_x >= win_xleft_low) & 
                                 (nonzero_x < win_xleft_high)).nonzero()[0]
                
                if len(good_left_inds) > minpix:
                    leftx_current = int(np.mean(nonzero_x[good_left_inds]))
                    points = np.column_stack((nonzero_x[good_left_inds], 
                                            nonzero_y[good_left_inds] + y_start))
                    left_points.extend(points)
        
        # Process right lane if base found
        if right_base is not None:
            rightx_current = right_base
            for window in range(n_windows):
                win_y_low = binary_img.shape[0] - (window + 1) * window_height
                win_y_high = binary_img.shape[0] - window * window_height
                
                win_xright_low = max(0, rightx_current - margin)
                win_xright_high = min(binary_img.shape[1], rightx_current + margin)
                
                good_right_inds = ((nonzero_y >= win_y_low) & 
                                  (nonzero_y < win_y_high) & 
                                  (nonzero_x >= win_xright_low) & 
                                  (nonzero_x < win_xright_high)).nonzero()[0]
                
                if len(good_right_inds) > minpix:
                    rightx_current = int(np.mean(nonzero_x[good_right_inds]))
                    points = np.column_stack((nonzero_x[good_right_inds], 
                                            nonzero_y[good_right_inds] + y_start))
                    right_points.extend(points)
        
        # Convert lists to numpy arrays
        left_points = np.array(left_points) if left_points else np.array([])
        right_points = np.array(right_points) if right_points else np.array([])
        
        # Determine detection status
        detection_status = 'none'
        if len(left_points) > minpix and len(right_points) > minpix:
            detection_status = 'both'
        elif len(left_points) > minpix:
            detection_status = 'left'
        elif len(right_points) > minpix:
            detection_status = 'right'
        
        return left_points, right_points, detection_status

    def _fit_curve(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Simple curve fitting using polynomial."""
        if not isinstance(points, np.ndarray) or len(points) < self.config['min_points_for_curve']:
            return None
        
        try:
            # Fit a second-degree polynomial
            coeffs = np.polyfit(points[:, 1], points[:, 0], 2)
            
            # Generate points along the curve
            y_points = np.linspace(
                np.min(points[:, 1]),
                np.max(points[:, 1]),
                num=20
            )
            x_points = np.polyval(coeffs, y_points)
            
            # Create curve points
            curve_points = np.column_stack((
                x_points.astype(np.int32),
                y_points.astype(np.int32)
            ))
            
            return curve_points
            
        except Exception as e:
            print(f"Warning: Failed to fit curve: {str(e)}")
            return None

    def _calculate_curve_radius(self, curve_points: np.ndarray) -> Optional[float]:
        """Calculate the radius of curvature for a lane curve.
        
        For partial lanes or limited FOV scenarios, we focus on calculating
        the curve radius using the visible portion of the lane.
        """
        if curve_points is None or len(curve_points) < 3:
            return None
            
        try:
            # Fit a polynomial to the visible portion of the curve
            coeffs = np.polyfit(curve_points[:, 1], curve_points[:, 0], 2)
            
            # Calculate radius at the bottom of the visible portion
            y_eval = np.max(curve_points[:, 1])
            dx_dy = 2 * coeffs[0] * y_eval + coeffs[1]
            d2x_dy2 = 2 * coeffs[0]
            
            radius = ((1 + dx_dy**2)**(3/2)) / abs(d2x_dy2) if d2x_dy2 != 0 else None
            return radius
            
        except:
            return None

    def _estimate_curve_direction(self, left_curve: Optional[np.ndarray], 
                                right_curve: Optional[np.ndarray]) -> float:
        """Estimate curve direction from available lane data.
        
        This method works with either or both lanes visible, adapting its
        calculations based on which lane information is available.
        """
        # Use whichever lane is visible, or average both if we have them
        curves_to_check = []
        if left_curve is not None:
            curves_to_check.append(left_curve)
        if right_curve is not None:
            curves_to_check.append(right_curve)
            
        if not curves_to_check:
            return self.prev_curve_direction
            
        # Calculate direction for each available curve
        directions = []
        for curve in curves_to_check:
            if len(curve) >= 2:
                # Calculate the overall trend of the curve
                top_idx = 0
                bottom_idx = -1
                dx = curve[top_idx][0] - curve[bottom_idx][0]
                dy = curve[top_idx][1] - curve[bottom_idx][1]
                
                # Normalize to get a direction value between -1 and 1
                max_dx = self.width * 0.3
                direction = dx / max_dx
                directions.append(direction)
        
        if not directions:
            return self.prev_curve_direction
            
        # Average the directions we found
        avg_direction = np.mean(directions)
        
        # Apply smoothing to prevent sudden changes
        smoothed_direction = (self.config['curve_smoothing_factor'] * 
                            self.prev_curve_direction +
                            (1 - self.config['curve_smoothing_factor']) * 
                            avg_direction)
        
        self.prev_curve_direction = smoothed_direction
        return smoothed_direction

    def process_frame(self, frame: DetectionFrame) -> DetectionResult:
        """Process a frame to detect and analyze lane curves.
        
        This method implements the complete lane detection pipeline with
        support for:
        - Limited FOV scenarios
        - Single-lane detection and tracking
        - Adaptive thresholding for varying lighting
        - Curve radius estimation from partial lanes
        """
        # Preprocess frame
        binary = self._preprocess_frame(frame.frame)
        roi = self._apply_roi(binary)
        
        # Find lane points and detection status
        left_points, right_points, detection_status = self._find_lane_points(roi)
        
        # Fit curves to detected points
        left_curve = self._fit_curve(left_points) if len(left_points) > 0 else None
        right_curve = self._fit_curve(right_points) if len(right_points) > 0 else None
        
        # Update lane width tracking
        estimated_width = self._update_lane_width(left_curve, right_curve)
        
        # Handle single lane cases with high confidence
        if estimated_width is not None and self.lane_width_confidence > 0.7:
            if detection_status == 'left' and left_curve is not None:
                # Estimate right lane position
                right_curve = left_curve + [estimated_width, 0]
            elif detection_status == 'right' and right_curve is not None:
                # Estimate left lane position
                left_curve = right_curve - [estimated_width, 0]
        
        # Calculate metrics based on available lanes
        if left_curve is not None or right_curve is not None:
            # Calculate center offset based on available information
            if left_curve is not None and right_curve is not None:
                lane_center = (left_curve[-1][0] + right_curve[-1][0]) // 2
            elif left_curve is not None:
                lane_center = left_curve[-1][0] + (estimated_width or 0) // 2
            else:  # right_curve is not None
                lane_center = right_curve[-1][0] - (estimated_width or 0) // 2
            
            frame_center = self.width // 2
            center_offset = lane_center - frame_center
            
            # Calculate curve metrics from available lane(s)
            reference_curve = left_curve if left_curve is not None else right_curve
            curve_radius = self._calculate_curve_radius(reference_curve)
            curve_direction = self._estimate_curve_direction(left_curve, right_curve)
            
            is_valid = True
        else:
            center_offset = 0
            curve_radius = None
            curve_direction = self.prev_curve_direction
            is_valid = False
        
        # Create lane detection data
        lane_data = LaneDetectionData(
            left_curve=left_curve,
            right_curve=right_curve,
            center_offset=center_offset,
            curve_radius=curve_radius,
            curve_direction=curve_direction
        )
        
        # Store comprehensive debug information
        metadata = {
            'binary_frame': roi,
            'roi_vertices': self.roi_vertices,
            'left_points': left_points if len(left_points) > 0 else None,
            'right_points': right_points if len(right_points) > 0 else None,
            'detection_status': detection_status,
            'lane_width_confidence': self.lane_width_confidence,
            'estimated_lane_width': estimated_width,
            'adaptive_threshold': self.config['adaptive_block_size'],
            'binary_frame_full': binary  # Full binary frame before ROI
        }
        
        return DetectionResult(
            is_valid=is_valid,
            data=lane_data,
            metadata=metadata
        )

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration parameters."""
        return self.config.copy()

    def reset(self):
        """Reset detector state.
        
        Clears all stateful data including:
        - Previous curve information
        - Lane width history
        - Confidence measures
        """
        self.prev_left_curve = None
        self.prev_right_curve = None
        self.prev_curve_radius = None
        self.prev_curve_direction = 0
        self.recent_lane_widths = []
        self.lane_width_confidence = 0.0