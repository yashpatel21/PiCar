import cv2
import numpy as np
import threading
from queue import Queue
from picamera2 import Picamera2
import time
from typing import Optional, Tuple, Callable

class CameraManager:
    """Manages camera operations and provides frame access to other components.
    
    This class handles camera initialization, frame capture, and distribution to 
    multiple consumers through a thread-safe queue system. It supports both
    synchronous and asynchronous frame access patterns.
    """
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        """Initialize camera with specified parameters.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Target frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        
        # Initialize camera with RGB888 format (provides BGR-ordered pixels)
        self.camera = Picamera2()
        self.camera_config = self.camera.create_preview_configuration(
            main={"format": 'RGB888', "size": (width, height)}
        )
        self.camera.configure(self.camera_config)
        
        # Frame distribution system
        self.frame_queue = Queue(maxsize=1)  # Latest frame only
        self.subscribers = []  # Callback functions for frame updates
        self.running = False
        self.capture_thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start camera capture and frame distribution."""
        if self.running:
            return
            
        self.camera.start()
        time.sleep(2)  # Allow camera to initialize
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
    def stop(self):
        """Stop camera capture and cleanup resources."""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()
        self.camera.stop()
        
    def _capture_loop(self):
        """Main capture loop - runs in separate thread."""
        last_frame_time = time.time()
        frame_interval = 1.0 / self.fps
        
        while self.running:
            current_time = time.time()
            if current_time - last_frame_time >= frame_interval:
                frame = self.camera.capture_array()  # BGR format
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Update frame queue
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(frame)
                
                # Notify subscribers
                for callback in self.subscribers:
                    callback(frame)
                    
                last_frame_time = current_time
                
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame synchronously.
        
        Returns:
            BGR format numpy array or None if no frame available
        """
        try:
            return self.frame_queue.get_nowait()
        except:
            return None
            
    def subscribe(self, callback: Callable[[np.ndarray], None]):
        """Register a callback for frame updates.
        
        Args:
            callback: Function to call with each new frame
        """
        self.subscribers.append(callback)
        
    def unsubscribe(self, callback: Callable[[np.ndarray], None]):
        """Remove a frame update callback."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)