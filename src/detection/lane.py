# src/detection/lane.py

import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict, Any
from .base import Detector, DetectionResult
from utils.types import DetectionFrame, LaneDetectionData
import time

class LaneDetector(Detector):
    """Enhanced lane detection system for limited FOV and single-line scenarios.
    
    This detector is specifically designed to work with:
    1. A camera with limited field of view where lanes may start partway up the frame
    2. Single black line detection for miniature track environments
    3. Sharp curves where lanes may partially leave the frame
    4. Variable lighting conditions through adaptive thresholding
    
    The detector now properly handles thick single lines by:
    - Using enhanced peak detection to identify line edges
    - Validating detected lines to prevent false dual-line detection
    - Implementing improved point tracking for single lines
    """
    
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        """Initialize the lane detector with given frame dimensions.
        
        Args:
            frame_width: Width of input frames in pixels
            frame_height: Height of input frames in pixels
        """
        self.width = frame_width
        self.height = frame_height
        
        # Define ROI from 40% to 80% of frame height for optimal detection
        y_start = int(self.height * 0.4)
        y_end = int(self.height * 0.8)
        
        self.roi_vertices = np.array([
            [(0, y_start),
             (0, y_end),
             (self.width, y_end),
             (self.width, y_start)]
        ], dtype=np.int32)
        
        # Configuration parameters with enhanced line detection settings
        self.config = {
            # Basic image processing parameters
            'blur_kernel_size': 7,       # Kernel size for noise reduction
            'adaptive_block_size': 45,    # Block size for adaptive threshold
            'adaptive_offset': 25,        # Constant subtracted from mean
            'dilate_iterations': 1,       # Number of dilation steps
            'erode_iterations': 1,        # Number of erosion steps
            
            # Curve fitting parameters
            'min_points_for_curve': 4,    # Minimum points needed for curve fitting
            'curve_smoothing_factor': 0.9, # Smoothing factor for curve direction
            
            # Enhanced line detection parameters
            'min_peak_value': 20,         # Minimum height for histogram peaks
            'min_peak_distance': 30,      # Minimum distance between distinct peaks
            'line_thickness_threshold': 50,# Maximum thickness for single line
            'window_height': 9,           # Number of sliding windows
            'window_margin': 60,          # Sliding window width (each side)
            'minpix': 15                  # Minimum pixels for window recentering
        }
        
        # State tracking for temporal smoothing
        self.prev_left_curve = None
        self.prev_right_curve = None
        self.prev_curve_radius = None
        self.prev_curve_direction = 0
        
        # Lane width learning system
        self.recent_lane_widths = []
        self.max_width_history = 10
        self.lane_width_confidence = 0.0
        
        # Pre-allocate arrays for curve fitting
        self.y_points = np.linspace(frame_height, 0, num=20)
        self.curve_points = np.zeros((20, 2), dtype=np.int32)

    def update_config(self, param: str, value: int) -> None:
        """Update a configuration parameter with validation.
        
        Args:
            param: Name of parameter to update
            value: New value for parameter
        """
        valid_ranges = {
            'blur_kernel_size': (3, 11),
            'adaptive_block_size': (3, 99),
            'adaptive_offset': (0, 50),
            'dilate_iterations': (0, 3),
            'erode_iterations': (0, 3),
            'min_points_for_curve': (3, 10),
            'min_peak_value': (10, 100),
            'min_peak_distance': (10, 100),
            'line_thickness_threshold': (20, 100),
            'window_margin': (30, 100),
            'minpix': (5, 50)
        }
        
        if param in valid_ranges:
            min_val, max_val = valid_ranges[param]
            validated_value = max(min_val, min(value, max_val))
            
            # Ensure odd values for kernel sizes
            if param in ['blur_kernel_size', 'adaptive_block_size']:
                validated_value = validated_value if validated_value % 2 == 1 else validated_value - 1
                
            self.config[param] = validated_value

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Prepare frame using adaptive thresholding for better lane isolation.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Binary image with isolated lane lines
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur for noise reduction
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
        
        # Clean up binary image with morphological operations
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.dilate(binary, kernel, 
                           iterations=self.config['dilate_iterations'])
        binary = cv2.erode(binary, kernel, 
                           iterations=self.config['erode_iterations'])
        
        return binary

    def _apply_roi(self, img: np.ndarray) -> np.ndarray:
        """Apply region of interest mask focusing on the relevant portion of the frame.
        
        Args:
            img: Input binary image
            
        Returns:
            Masked image containing only the ROI
        """
        # Calculate ROI boundaries
        y_start = int(self.height * 0.4)
        y_end = int(self.height * 0.8)
        roi_height = y_end - y_start
        
        # Create mask for ROI area
        mask = np.zeros((roi_height, self.width), dtype=img.dtype)
        
        # Adjust vertices for the mask
        roi_vertices = self.roi_vertices.copy()
        roi_vertices[:, :, 1] -= y_start
        
        # Apply mask
        cv2.fillPoly(mask, roi_vertices, 255)
        return cv2.bitwise_and(img[y_start:y_end], mask)

    def _find_peaks(self, histogram: np.ndarray) -> List[int]:
        """Find significant peaks in the histogram that represent potential lane lines.
        
        Args:
            histogram: Input histogram of white pixels
            
        Returns:
            List of peak x-coordinates
        """
        peaks = []
        window_size = 5
        min_peak_value = self.config['min_peak_value']
        
        # Slide window across histogram to find local maxima
        for i in range(window_size, len(histogram) - window_size):
            window = histogram[i-window_size:i+window_size]
            if (histogram[i] > min_peak_value and 
                histogram[i] == max(window) and
                histogram[i] > histogram[i-1] and 
                histogram[i] > histogram[i+1]):
                peaks.append(i)
        
        return peaks

    def _merge_peaks(self, peaks: List[int]) -> List[int]:
        """Merge peaks that are likely from the same line.
        
        Args:
            peaks: List of detected peak x-coordinates
            
        Returns:
            List of merged peak x-coordinates
        """
        merged_peaks = []
        min_peak_distance = self.config['min_peak_distance']
        
        for peak in sorted(peaks):
            if not merged_peaks or abs(peak - merged_peaks[-1]) > min_peak_distance:
                merged_peaks.append(peak)
            else:
                # Update existing peak position to average
                merged_peaks[-1] = (merged_peaks[-1] + peak) // 2
        
        return merged_peaks

    def _track_line_points(self, binary_img: np.ndarray, base_x: int,
                          nonzero_x: np.ndarray, nonzero_y: np.ndarray) -> List:
        """Track points along a single line using sliding windows.
        
        Args:
            binary_img: Binary input image
            base_x: Starting x-coordinate for tracking
            nonzero_x: x-coordinates of white pixels
            nonzero_y: y-coordinates of white pixels
            
        Returns:
            List of points belonging to the tracked line
        """
        points = []
        current_x = base_x
        
        window_height = binary_img.shape[0] // self.config['window_height']
        margin = self.config['window_margin']
        minpix = self.config['minpix']
        
        # Slide window from bottom to top
        for window in range(self.config['window_height']):
            win_y_low = binary_img.shape[0] - (window + 1) * window_height
            win_y_high = binary_img.shape[0] - window * window_height
            
            win_x_low = max(0, current_x - margin)
            win_x_high = min(binary_img.shape[1], current_x + margin)
            
            # Find points in window
            good_inds = ((nonzero_y >= win_y_low) & 
                        (nonzero_y < win_y_high) & 
                        (nonzero_x >= win_x_low) & 
                        (nonzero_x < win_x_high)).nonzero()[0]
            
            if len(good_inds) > minpix:
                current_x = int(np.mean(nonzero_x[good_inds]))
                points.extend(np.column_stack((nonzero_x[good_inds], 
                                            nonzero_y[good_inds])))
        
        return points

    def _find_lane_points(self, binary_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """Enhanced lane point detection that properly handles both straight and curved sections.
        
        This method implements a sophisticated lane detection approach that:
        1. Identifies potential lane lines from the binary image
        2. Validates detections using geometric properties
        3. Handles both single and dual lane scenarios
        4. Prevents false detections from thick lines
        
        Args:
            binary_img: Thresholded binary image where white pixels represent potential lane lines
            
        Returns:
            Tuple containing:
            - left_points: numpy array of points for left lane (empty if not detected)
            - right_points: numpy array of points for right lane (empty if not detected)
            - detection_status: string indicating what was detected ('left', 'right', 'both', or 'none')
        """
        # Get the starting y-coordinate for our ROI
        y_start = int(self.height * 0.4)
        
        # Find all white pixels in the binary image
        nonzero_y, nonzero_x = binary_img.nonzero()
        
        # Take histogram of bottom half of the image to find starting points
        histogram = np.sum(binary_img[binary_img.shape[0]//2:, :], axis=0)
        midpoint = binary_img.shape[1] // 2
        
        # Split histogram into left and right halves
        left_hist = histogram[:midpoint]
        right_hist = histogram[midpoint:]
        
        # Get the strength of peaks in each half
        left_max = np.max(left_hist)
        right_max = np.max(right_hist)
        
        # Parameters for lane detection
        min_peak_value = 20          # Minimum peak height to be considered a lane
        min_peak_ratio = 0.4         # Weaker peak must be at least 40% of stronger peak
        window_height = binary_img.shape[0] // 9  # Height of search windows
        margin = 80                  # Width of search windows
        minpix = 15                  # Minimum pixels needed to recenter window
        
        # Initialize lane base points
        left_base = None
        right_base = None
        
        # Only accept peaks that are strong enough
        if left_max > min_peak_value:
            left_base = np.argmax(left_hist)
            
        if right_max > min_peak_value:
            right_base = midpoint + np.argmax(right_hist)
        
        # If we found both peaks, validate their relative strengths
        if left_base is not None and right_base is not None:
            peak_ratio = min(left_max, right_max) / max(left_max, right_max)
            if peak_ratio < min_peak_ratio:
                # Peaks are too different in strength - likely one is noise
                # Keep only the stronger peak
                if left_max > right_max:
                    right_base = None
                else:
                    left_base = None
        
        # Lists to store detected lane points
        left_points = []
        right_points = []
        
        # Track left lane points if we found a valid starting point
        if left_base is not None:
            current_x = left_base
            
            # Use sliding windows moving upward through the image
            for window in range(9):
                # Calculate window boundaries
                win_y_low = binary_img.shape[0] - (window + 1) * window_height
                win_y_high = binary_img.shape[0] - window * window_height
                win_x_low = max(0, current_x - margin)
                win_x_high = min(binary_img.shape[1], current_x + margin)
                
                # Find points in the window
                good_inds = ((nonzero_y >= win_y_low) & 
                            (nonzero_y < win_y_high) & 
                            (nonzero_x >= win_x_low) & 
                            (nonzero_x < win_x_high)).nonzero()[0]
                
                # If we found enough points, recenter the window
                if len(good_inds) > minpix:
                    current_x = int(np.mean(nonzero_x[good_inds]))
                    # Add points to our list
                    points = np.column_stack((nonzero_x[good_inds], 
                                            nonzero_y[good_inds] + y_start))
                    left_points.extend(points)
        
        # Track right lane points with the same process
        if right_base is not None:
            current_x = right_base
            
            for window in range(9):
                win_y_low = binary_img.shape[0] - (window + 1) * window_height
                win_y_high = binary_img.shape[0] - window * window_height
                win_x_low = max(0, current_x - margin)
                win_x_high = min(binary_img.shape[1], current_x + margin)
                
                good_inds = ((nonzero_y >= win_y_low) & 
                            (nonzero_y < win_y_high) & 
                            (nonzero_x >= win_x_low) & 
                            (nonzero_x < win_x_high)).nonzero()[0]
                
                if len(good_inds) > minpix:
                    current_x = int(np.mean(nonzero_x[good_inds]))
                    points = np.column_stack((nonzero_x[good_inds], 
                                            nonzero_y[good_inds] + y_start))
                    right_points.extend(points)
        
        # Convert lists to numpy arrays
        left_points = np.array(left_points) if left_points else np.array([])
        right_points = np.array(right_points) if right_points else np.array([])
        
        # If we found points for both lanes, validate their geometric relationship
        if len(left_points) > minpix and len(right_points) > minpix:
            # Fit preliminary curves to check geometry
            left_curve = self._fit_curve(left_points)
            right_curve = self._fit_curve(right_points)
            
            if not self._validate_curves(left_curve, right_curve):
                # Invalid geometry - keep only the lane with more points
                if len(left_points) > len(right_points):
                    right_points = np.array([])
                else:
                    left_points = np.array([])
        
        # Determine final detection status
        if len(left_points) > minpix and len(right_points) > minpix:
            detection_status = 'both'
        elif len(left_points) > minpix:
            detection_status = 'left'
        elif len(right_points) > minpix:
            detection_status = 'right'
        else:
            detection_status = 'none'
        
        return left_points, right_points, detection_status

    def _validate_curves(self, left_curve: np.ndarray, right_curve: np.ndarray) -> bool:
        """Validates detected curves based on geometric properties and relationships.
        
        Uses the fundamental geometry of road lanes to determine if detected curves
        could represent real lanes. Key principle: when curves go in the same direction,
        they must maintain a significant separation to be considered separate lanes.
        """
        if left_curve is None or right_curve is None:
            return True
            
        # Constants for validation
        min_expected_separation = 50  # Minimum pixels between valid lanes
        min_direction_diff = 0.3      # Minimum difference in direction vectors
        
        # Get direction vectors (from bottom to top of curves)
        def get_curve_direction(curve):
            if len(curve) < 2:
                return None
                
            # Use multiple points to get a more stable direction
            bottom_section = curve[-3:]  # Last 3 points
            top_section = curve[:3]      # First 3 points
            
            if len(bottom_section) < 1 or len(top_section) < 1:
                return None
                
            # Calculate average points
            bottom_x = np.mean(bottom_section[:, 0])
            bottom_y = np.mean(bottom_section[:, 1])
            top_x = np.mean(top_section[:, 0])
            top_y = np.mean(top_section[:, 1])
            
            # Get direction vector
            dx = top_x - bottom_x
            dy = top_y - bottom_y
            
            # Normalize the vector
            magnitude = np.sqrt(dx*dx + dy*dy)
            if magnitude == 0:
                return None
                
            return dx/magnitude, dy/magnitude
        
        # Get directions for both curves
        left_dir = get_curve_direction(left_curve)
        right_dir = get_curve_direction(right_curve)
        
        if left_dir is None or right_dir is None:
            return False
        
        # Calculate how parallel the curves are
        dot_product = left_dir[0]*right_dir[0] + left_dir[1]*right_dir[1]
        
        # Calculate separation between curves
        separations = []
        for i in range(min(len(left_curve), len(right_curve))):
            separation = abs(right_curve[i][0] - left_curve[i][0])
            separations.append(separation)
        
        if not separations:
            return False
            
        avg_separation = np.mean(separations)
        
        # Key validation logic:
        # 1. If both curves are going right (positive x direction)
        if left_dir[0] > 0 and right_dir[0] > 0:
            # They must be well-separated to be valid lanes
            if avg_separation < min_expected_separation:
                return False
            # If they're very parallel (similar direction), require even more separation
            if dot_product > 0.9:  # Very parallel
                return avg_separation > min_expected_separation * 1.5
        
        # 2. If both curves are going left (negative x direction)
        if left_dir[0] < 0 and right_dir[0] < 0:
            if avg_separation < min_expected_separation:
                return False
            if dot_product > 0.9:
                return avg_separation > min_expected_separation * 1.5
        
        return True

    def _validate_detection(self, left_points: np.ndarray, 
                          right_points: np.ndarray) -> bool:
        """Validate that detected lines are actually separate lanes.
        
        Args:
            left_points: Points detected as left lane
            right_points: Points detected as right lane
            
        Returns:
            True if lines are distinct, False if likely same line
        """
        if len(left_points) == 0 or len(right_points) == 0:
            return True
            
        min_separation = float('inf')
        for left_point in left_points:
            for right_point in right_points:
                if abs(left_point[1] - right_point[1]) < 10:
                    separation = abs(left_point[0] - right_point[0])
                    min_separation = min(min_separation, separation)
        
        return min_separation > self.config['line_thickness_threshold']

    def _fit_curve(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Fit a smooth curve to detected lane points using polynomial regression.
        
        The method fits a second-degree polynomial to the points to create a 
        smooth curve representation of the lane line. This is especially important
        for single-line tracking where we need precise curve estimation.
        
        Args:
            points: Array of detected lane points
            
        Returns:
            Array of points along the fitted curve, or None if fitting fails
        """
        if not isinstance(points, np.ndarray) or len(points) < self.config['min_points_for_curve']:
            return None
        
        try:
            # Fit second-degree polynomial to points
            coeffs = np.polyfit(points[:, 1], points[:, 0], 2)
            
            # Generate smooth curve points
            y_points = np.linspace(
                np.min(points[:, 1]),
                np.max(points[:, 1]),
                num=20
            )
            x_points = np.polyval(coeffs, y_points)
            
            # Create curve point array
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
        
        This calculation is crucial for determining how sharp a turn is, which
        affects steering decisions. For single-line tracking, this becomes even
        more important as it's our primary indicator of turn severity.
        
        Args:
            curve_points: Points along the fitted curve
            
        Returns:
            Radius of curvature in pixels, or None if calculation fails
        """
        if curve_points is None or len(curve_points) < 3:
            return None
            
        try:
            # Fit polynomial to get coefficients for radius calculation
            coeffs = np.polyfit(curve_points[:, 1], curve_points[:, 0], 2)
            
            # Calculate radius at the bottom point of the curve
            y_eval = np.max(curve_points[:, 1])
            dx_dy = 2 * coeffs[0] * y_eval + coeffs[1]
            d2x_dy2 = 2 * coeffs[0]
            
            # Calculate radius using the curve formula
            if d2x_dy2 != 0:
                radius = ((1 + dx_dy**2)**(3/2)) / abs(d2x_dy2)
                return radius
            return None
            
        except:
            return None

    def _estimate_curve_direction(self, left_curve: Optional[np.ndarray], 
                                right_curve: Optional[np.ndarray]) -> float:
        """Estimate curve direction from available lane data.
        
        This method has been enhanced to work particularly well with single-line
        scenarios, providing stable direction estimation even when only one line
        is visible.
        
        Args:
            left_curve: Points of left lane curve if detected
            right_curve: Points of right lane curve if detected
            
        Returns:
            Direction value between -1 (left) and 1 (right)
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
                # Calculate curve direction using endpoints
                dx = curve[0][0] - curve[-1][0]
                dy = curve[0][1] - curve[-1][1]
                
                # Normalize direction value
                max_dx = self.width * 0.3
                direction = dx / max_dx
                directions.append(direction)
        
        if not directions:
            return self.prev_curve_direction
            
        # Calculate average direction and apply smoothing
        avg_direction = np.mean(directions)
        smoothed_direction = (self.config['curve_smoothing_factor'] * 
                            self.prev_curve_direction +
                            (1 - self.config['curve_smoothing_factor']) * 
                            avg_direction)
        
        self.prev_curve_direction = smoothed_direction
        return smoothed_direction

    def process_frame(self, frame: DetectionFrame) -> DetectionResult:
        """Process a frame to detect and analyze lane curves.
        
        This method implements the complete lane detection pipeline with
        support for thick single-line tracking. It handles preprocessing,
        detection, curve fitting, and metric calculation.
        
        Args:
            frame: A DetectionFrame object containing the input frame and metadata
            
        Returns:
            DetectionResult containing lane curves, metrics, and debug information
        """
        # First, preprocess the frame to isolate potential lane lines
        binary = self._preprocess_frame(frame.frame)
        roi = self._apply_roi(binary)
        
        # Detect lane points with our simplified single-line approach
        left_points, right_points, detection_status = self._find_lane_points(roi)
        
        # Fit curves to the detected points if we have them
        left_curve = self._fit_curve(left_points) if len(left_points) > 0 else None
        right_curve = self._fit_curve(right_points) if len(right_points) > 0 else None
        
        # Calculate metrics if we detected any line
        if left_curve is not None or right_curve is not None:
            # Calculate center offset based on which line we detected
            if left_curve is not None:
                # For left line, assume the lane center is a fixed width to the right
                lane_center = left_curve[-1][0] + self.width // 4
            else:
                # For right line, assume the lane center is a fixed width to the left
                lane_center = right_curve[-1][0] - self.width // 4
            
            # Calculate offset from frame center
            frame_center = self.width // 2
            center_offset = lane_center - frame_center
            
            # Get the curve we detected for radius and direction calculations
            reference_curve = left_curve if left_curve is not None else right_curve
            curve_radius = self._calculate_curve_radius(reference_curve)
            curve_direction = self._estimate_curve_direction(left_curve, right_curve)
            
            # Mark detection as valid since we found a line
            is_valid = True
        else:
            # No lines detected, use default/fallback values
            center_offset = 0
            curve_radius = None
            curve_direction = self.prev_curve_direction
            is_valid = False
        
        # Create lane detection data structure with our results
        lane_data = LaneDetectionData(
            left_curve=left_curve,
            right_curve=right_curve,
            center_offset=center_offset,
            curve_radius=curve_radius,
            curve_direction=curve_direction
        )
        
        # Store comprehensive debug information for visualization
        metadata = {
            'binary_frame': roi,                   # ROI with binary threshold applied
            'roi_vertices': self.roi_vertices,     # ROI boundary points
            'left_points': left_points if len(left_points) > 0 else None,
            'right_points': right_points if len(right_points) > 0 else None,
            'detection_status': detection_status,  # Whether we found left/right/both lines
            'lane_width_confidence': self.lane_width_confidence,
            'adaptive_threshold': self.config['adaptive_block_size'],
            'binary_frame_full': binary,          # Full binary frame before ROI
            'processing_time': time.time() - frame.timestamp  # Performance metric
        }
        
        # Return the complete detection result
        return DetectionResult(
            is_valid=is_valid,
            data=lane_data,
            metadata=metadata
        )

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration parameters."""
        return self.config.copy()

    def reset(self):
        """Reset detector state and clear history."""
        self.prev_left_curve = None
        self.prev_right_curve = None
        self.prev_curve_radius = None
        self.prev_curve_direction = 0
        self.recent_lane_widths = []
        self.lane_width_confidence = 0.0