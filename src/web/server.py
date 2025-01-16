# src/web/server.py

from flask import Flask, render_template, jsonify, Response
from flask_socketio import SocketIO, emit
import threading
from queue import Queue
import cv2
import numpy as np
import base64
import time
from utils.types import VisualizationResult
from typing import Dict, Optional, Any

class StreamManager:
    """Manages high-performance video streaming using WebSocket technology.
    
    This class handles efficient frame distribution to web clients while
    maintaining high frame rates and good image quality. It uses techniques like:
    - Efficient frame encoding with optimized JPEG compression
    - Frame rate limiting to prevent overload
    - Non-blocking operations for smooth performance
    """
    
    def __init__(self):
        # Initialize stream buffers for different views
        self.streams = {
            'main': Queue(maxsize=2),          # Main driving view
            'binary': Queue(maxsize=2),        # Binary threshold view
            'roi': Queue(maxsize=2),           # Region of interest view
            'points': Queue(maxsize=2)         # Point detection view
        }
        
        # Thread synchronization
        self.locks = {name: threading.Lock() for name in self.streams}
        
        # Performance optimization settings
        self.jpeg_quality = 80        # Balance between quality and speed
        self.max_fps = 30            # Target maximum frame rate
        self.min_frame_time = 1.0 / self.max_fps
        
        # Stream state tracking
        self.active_streams = set(['main'])
        self.last_frame_times = {name: 0.0 for name in self.streams}
        self.stream_fps = {name: 0.0 for name in self.streams}
        
        # FPS calculation
        self.frame_times = {name: [] for name in self.streams}
        self.max_frame_history = 30

    def update_frames(self, viz_result: VisualizationResult):
        """Update all active stream buffers with new visualization data.
        
        Uses efficient frame encoding and rate limiting to maintain
        optimal performance.
        """
        current_time = time.time()
        
        # Update main view - always active
        self._update_stream('main', viz_result.main_frame, current_time)
        
        # Update debug views if enabled
        if viz_result.debug_enabled and viz_result.debug_views:
            for view_name, frame in viz_result.debug_views.items():
                if view_name in self.active_streams:
                    self._update_stream(view_name, frame, current_time)

    def _update_stream(self, stream_name: str, frame: np.ndarray, current_time: float):
        """Efficiently update a single stream's frame buffer."""
        # Check frame rate limiting
        if current_time - self.last_frame_times[stream_name] < self.min_frame_time:
            return
            
        with self.locks[stream_name]:
            try:
                # Update FPS calculation
                frame_time = current_time - self.last_frame_times[stream_name]
                self.frame_times[stream_name].append(frame_time)
                if len(self.frame_times[stream_name]) > self.max_frame_history:
                    self.frame_times[stream_name].pop(0)
                
                # Calculate average FPS
                if self.frame_times[stream_name]:
                    avg_frame_time = sum(self.frame_times[stream_name]) / len(self.frame_times[stream_name])
                    self.stream_fps[stream_name] = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
                
                # Update frame processing
                frame_rgb = frame[..., ::-1]
                encode_params = [
                    int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality,
                    int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
                ]
                _, buffer = cv2.imencode('.jpg', frame_rgb, encode_params)
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                if self.streams[stream_name].full():
                    try:
                        self.streams[stream_name].get_nowait()
                    except Queue.Empty:
                        pass
                self.streams[stream_name].put_nowait(frame_data)
                
                self.last_frame_times[stream_name] = current_time
                
            except Exception as e:
                print(f"Error updating stream {stream_name}: {str(e)}")

    def get_frame(self, stream_name: str) -> Optional[str]:
        """Get the latest frame from a stream.
        
        Returns base64-encoded JPEG data ready for WebSocket transmission.
        """
        if stream_name not in self.active_streams and stream_name != 'main':
            return None
            
        with self.locks[stream_name]:
            try:
                return self.streams[stream_name].get_nowait()
            except:
                return None

    def get_stream_info(self) -> Dict[str, Any]:
        """Get current status of all streams."""
        return {
            'active_streams': list(self.active_streams),
            'fps': {name: round(fps, 1) 
                   for name, fps in self.stream_fps.items()}
        }

class WebServer:
    """Flask-SocketIO based web interface for the self-driving system.
    
    Implements high-performance video streaming and real-time controls
    using WebSocket technology for minimal latency.
    """
    
    def __init__(self, system_instance, host='0.0.0.0', port=5000):
        # Initialize Flask and SocketIO
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, async_mode='threading', 
                               cors_allowed_origins='*')
        self.host = host
        self.port = port
        
        # Store system reference
        self.system = system_instance
        
        # Initialize stream management
        self.stream_manager = StreamManager()
        
        # Ensure main and binary streams are active by default
        self.stream_manager.active_streams = set(['main', 'binary'])
        
        # System state
        self.autonomous_enabled = False
        self.debug_enabled = True
        self.latest_metrics = None
        
        # Thread synchronization
        self.state_lock = threading.Lock()
        
        # Register routes and socket handlers
        self._register_routes()
        self._register_socket_handlers()

    def _register_routes(self):
        """Set up HTTP routes."""
        self.app.route('/')(self.index)
        self.app.route('/api/status')(self.get_status)
        self.app.route('/api/metrics')(self.get_metrics)
        
        # Add new video feed route
        @self.app.route('/video_feed/<stream_name>')
        def video_feed(stream_name):
            """Video streaming route."""
            if stream_name not in self.stream_manager.streams:
                return Response(status=404)
                
            def generate():
                while True:
                    frame_data = self.stream_manager.get_frame(stream_name)
                    if frame_data:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               base64.b64decode(frame_data) + b'\r\n')
            
            return Response(generate(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')

    def _register_socket_handlers(self):
        """Set up WebSocket event handlers."""
        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')
        
        @self.socketio.on('request_frame')
        def handle_frame_request(data):
            stream_name = data.get('stream')
            if stream_name in self.stream_manager.streams:
                # Only send frames for active streams or main view
                if stream_name == 'main' or stream_name in self.stream_manager.active_streams:
                    frame_data = self.stream_manager.get_frame(stream_name)
                    if frame_data is not None:
                        emit('frame_update', {
                            'stream': stream_name,
                            'frame': frame_data
                        })
        
        @self.socketio.on('toggle_autonomous')
        def handle_toggle_autonomous():
            """Handle autonomous mode toggle from web interface.
            
            Updates both the web server state and the main system state,
            then broadcasts the new status to all connected clients.
            """
            with self.state_lock:
                # Toggle the state
                self.autonomous_enabled = not self.autonomous_enabled
                
                # Update the main system's autonomous mode
                self.system.set_autonomous_mode(self.autonomous_enabled)
                
                # Broadcast the new state to all clients
                emit('autonomous_status', {
                    'autonomous_enabled': self.autonomous_enabled
                }, broadcast=True)
        
        @self.socketio.on('toggle_debug')
        def handle_toggle_debug():
            with self.state_lock:
                self.debug_enabled = not self.debug_enabled
                emit('status_update', {
                    'debug_enabled': self.debug_enabled
                })
        
        @self.socketio.on('toggle_stream')
        def handle_toggle_stream(data):
            stream_name = data.get('stream')
            print(f"Toggle stream request: {stream_name}")  # Debug log
            
            if stream_name in self.stream_manager.streams and stream_name != 'main':
                with self.state_lock:
                    if stream_name in self.stream_manager.active_streams:
                        print(f"Removing stream: {stream_name}")  # Debug log
                        self.stream_manager.active_streams.remove(stream_name)
                    else:
                        print(f"Adding stream: {stream_name}")  # Debug log
                        self.stream_manager.active_streams.add(stream_name)
                    
                    active_streams = list(self.stream_manager.active_streams)
                    print(f"Active streams: {active_streams}")  # Debug log
                    emit('stream_update', {
                        'active_streams': active_streams
                    }, broadcast=True)
        
        @self.socketio.on('update_config')
        def handle_config_update(data):
            """Handle configuration parameter updates."""
            try:
                param = data.get('param')
                value = int(data.get('value'))
                print(f"Updating config: {param} = {value}")  # Debug log
                
                # Update lane detector config
                self.system.lane_detector.update_config(param, value)
                
                # Get updated config and broadcast to all clients
                updated_config = self.system.lane_detector.get_config()
                emit('config_update', {
                    'success': True,
                    'config': updated_config
                }, broadcast=True)
                
            except Exception as e:
                print(f"Error updating config: {str(e)}")  # Debug log
                emit('config_update', {
                    'success': False,
                    'error': str(e)
                })

    def index(self):
        """Render main page."""
        return render_template('index.html')

    def update_visualization(self, viz_result: VisualizationResult):
        """Update all visualization streams with new data."""
        self.stream_manager.update_frames(viz_result)
        with self.state_lock:
            self.latest_metrics = viz_result.metrics

    def get_status(self):
        """Provide current system status."""
        with self.state_lock:
            return jsonify({
                'autonomous_enabled': self.autonomous_enabled,
                'debug_enabled': self.debug_enabled,
                'streams': self.stream_manager.get_stream_info(),
                'config': self.system.lane_detector.get_config()
            })

    def get_metrics(self):
        """Provide current performance metrics."""
        with self.state_lock:
            if self.latest_metrics:
                # Convert metrics to native Python types before serializing
                metrics_dict = {}
                for key, value in self.latest_metrics.to_dict().items():
                    # Convert numpy types to native Python types
                    if hasattr(value, 'item'):  # Check if it's a numpy type
                        metrics_dict[key] = value.item()
                    else:
                        metrics_dict[key] = value
                return jsonify(metrics_dict)
        return jsonify({})

    def run(self):
        """Start the Flask-SocketIO server."""
        self.socketio.run(self.app, host=self.host, port=self.port, 
                         allow_unsafe_werkzeug=True)