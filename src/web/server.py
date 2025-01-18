# src/web/server.py

from flask import Flask, render_template, jsonify, Response, request
from flask_socketio import SocketIO, emit
import threading
from queue import Queue
import cv2
import numpy as np
import base64
import time
from utils.types import VisualizationResult
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from werkzeug.wrappers import Response

class FrameBuffer:
    """Efficient circular buffer for managing video frames."""
    
    def __init__(self, max_size: int = 3):
        self.buffers = [None] * max_size
        self.write_index = 0
        self.read_index = 0
        self.lock = threading.Lock()
        self.last_write_time = 0.0
        self.frame_times = []
        self.max_frame_history = 30
        
    def write_frame(self, frame: np.ndarray) -> None:
        """Write a new frame to the buffer with base64 encoding."""
        current_time = time.time()
        
        with self.lock:
            # Convert frame to base64 string format
            if isinstance(frame, bytes):
                frame_base64 = base64.b64encode(frame).decode('utf-8')
            else:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            self.buffers[self.write_index] = frame_base64
            self.write_index = (self.write_index + 1) % len(self.buffers)
            
            if self.last_write_time > 0:
                frame_time = current_time - self.last_write_time
                self.frame_times.append(frame_time)
                if len(self.frame_times) > self.max_frame_history:
                    self.frame_times.pop(0)
            
            self.last_write_time = current_time
            
    def read_frame(self) -> Optional[str]:
        """Read the next available frame as a base64 string."""
        with self.lock:
            frame = self.buffers[self.read_index]
            if frame is not None:
                self.buffers[self.read_index] = None
                self.read_index = (self.read_index + 1) % len(self.buffers)
            return frame
            
    def get_fps(self) -> float:
        """Calculate current FPS based on frame timing history."""
        if not self.frame_times:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

class StreamManager:
    """Manages high-performance video streaming using WebSocket technology."""
    
    def __init__(self):
        self.streams = {
            'main': FrameBuffer(max_size=3),
            'binary': FrameBuffer(max_size=3),
            'roi': FrameBuffer(max_size=3),
            'points': FrameBuffer(max_size=3)
        }
        
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.jpeg_quality = 80
        self.max_fps = 30
        self.min_frame_time = 1.0 / self.max_fps
        self.active_streams = set(['main'])
        self.last_frame_times = {name: 0.0 for name in self.streams}
        
    def update_frames(self, viz_result: VisualizationResult) -> None:
        """Update all active stream buffers with new visualization data.
        
        Since frames are already in BGR format from the camera, we can process
        them directly without color space conversion. The frames flow through
        the system like this:
        1. Camera provides BGR frames
        2. Lane detection and visualization work with BGR
        3. JPEG compression expects BGR
        4. Web display shows colors correctly
        """
        current_time = time.time()
        
        # Process main view directly - no color conversion needed
        if viz_result.main_frame is not None:
            self._process_and_store_frame('main', viz_result.main_frame, current_time)
        
        # Process debug views in parallel if enabled
        if viz_result.debug_enabled and viz_result.debug_views:
            futures = []
            for view_name, frame in viz_result.debug_views.items():
                if view_name in self.active_streams and frame is not None:
                    futures.append(
                        self.thread_pool.submit(
                            self._process_and_store_frame,
                            view_name, frame, current_time
                        )
                    )
            for future in futures:
                future.result()

    def _process_and_store_frame(self, stream_name: str, frame: np.ndarray, 
                                current_time: float) -> None:
        """Process and store a single frame for a specific stream."""
        if current_time - self.last_frame_times[stream_name] < self.min_frame_time:
            return
            
        try:
            encode_params = [
                int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality,
                int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
            ]
            _, buffer = cv2.imencode('.jpg', frame, encode_params)
            
            self.streams[stream_name].write_frame(buffer.tobytes())
            self.last_frame_times[stream_name] = current_time
            
        except Exception as e:
            print(f"Error processing frame for {stream_name}: {str(e)}")
            
    def get_frame(self, stream_name: str) -> Optional[str]:
        """Get the latest frame from a specified stream."""
        if stream_name not in self.streams:
            return None
            
        if stream_name not in self.active_streams and stream_name != 'main':
            return None
            
        try:
            return self.streams[stream_name].read_frame()
        except Exception as e:
            print(f"Error reading frame from {stream_name}: {str(e)}")
            return None
        
    def get_stream_info(self) -> Dict[str, Any]:
        """Get current status and performance metrics for all streams."""
        try:
            return {
                'active_streams': list(self.active_streams),
                'fps': {name: round(stream.get_fps(), 1) 
                       for name, stream in self.streams.items()}
            }
        except Exception as e:
            print(f"Error getting stream info: {str(e)}")
            return {
                'active_streams': list(self.active_streams),
                'fps': {name: 0.0 for name in self.streams}
            }

class WebServer:
    """Flask-SocketIO based web interface for the self-driving system."""
    
    def __init__(self, system_instance, host='0.0.0.0', port=5000):
        self.app = Flask(__name__)
        self.app.config.update(
            SECRET_KEY='your-secret-key-here',
            DEBUG=False,
            USE_RELOADER=False,
            TEMPLATES_AUTO_RELOAD=False,
            MAX_CONTENT_LENGTH=50 * 1024 * 1024
        )
        
        self.socketio = SocketIO(
            self.app,
            async_mode='threading',
            cors_allowed_origins='*',
            logger=False,
            engineio_logger=False,
            ping_timeout=5,
            ping_interval=25,
            max_http_buffer_size=1e8,
            async_handlers=True,
            manage_session=False
        )
        
        self.host = host
        self.port = port
        self.system = system_instance
        self.stream_manager = StreamManager()
        self.stream_manager.active_streams = set(['main', 'binary'])
        
        self.autonomous_enabled = False
        self.debug_enabled = True
        self.latest_metrics = None
        
        self.state_lock = threading.Lock()
        self.active_clients = set()
        self.client_push_threads = {}
        
        self._register_routes()
        self._register_socket_handlers()
        
    def _register_routes(self):
        """Set up HTTP routes."""
        self.app.route('/')(self.index)
        self.app.route('/api/status')(self.get_status)
        self.app.route('/api/metrics')(self.get_metrics)
        
    def _register_socket_handlers(self):
        """Set up WebSocket event handlers."""
        
        @self.socketio.on_error_default
        def default_error_handler(e):
            print(f"SocketIO error occurred: {str(e)}")
            
        @self.socketio.on_error()
        def error_handler(e):
            print(f"SocketIO event handler error: {str(e)}")

        @self.socketio.on('connect_error')
        def connect_error_handler(e):
            print(f"SocketIO connection error: {str(e)}")
        
        @self.socketio.on('connect')
        def handle_connect():
            client_id = request.sid
            with self.state_lock:
                self.active_clients.add(client_id)
                self._start_frame_push(client_id)
            print(f'Client connected: {client_id}')
        
        @self.socketio.on('disconnect')
        def handle_disconnect(sid=None):
            try:
                client_id = request.sid
                with self.state_lock:
                    self.active_clients.discard(client_id)
                    if client_id in self.client_push_threads:
                        thread_data = self.client_push_threads[client_id]
                        thread_data['running'] = False
                        if thread_data['thread'].is_alive():
                            thread_data['thread'].join(timeout=0.5)
                        del self.client_push_threads[client_id]
                print(f'Client disconnected cleanly: {client_id}')
            except Exception as e:
                print(f'Error during client disconnect: {str(e)}')
        
        @self.socketio.on('toggle_autonomous')
        def handle_toggle_autonomous():
            with self.state_lock:
                self.autonomous_enabled = not self.autonomous_enabled
                self.system.set_autonomous_mode(self.autonomous_enabled)
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
            if stream_name in self.stream_manager.streams and stream_name != 'main':
                with self.state_lock:
                    if stream_name in self.stream_manager.active_streams:
                        self.stream_manager.active_streams.remove(stream_name)
                    else:
                        self.stream_manager.active_streams.add(stream_name)
                    emit('stream_update', {
                        'active_streams': list(self.stream_manager.active_streams)
                    }, broadcast=True)
        
        @self.socketio.on('update_config')
        def handle_config_update(data):
            try:
                param = data.get('param')
                value = int(data.get('value'))
                self.system.lane_detector.update_config(param, value)
                updated_config = self.system.lane_detector.get_config()
                emit('config_update', {
                    'success': True,
                    'config': updated_config
                }, broadcast=True)
            except Exception as e:
                emit('config_update', {
                    'success': False,
                    'error': str(e)
                })

    def _start_frame_push(self, client_id: str) -> None:
        """Start frame pushing thread for a specific client."""
        def push_frames():
            thread_data = self.client_push_threads[client_id]
            
            while thread_data['running']:
                try:
                    frames_to_send = {}
                    for stream_name in self.stream_manager.active_streams:
                        frame_data = self.stream_manager.get_frame(stream_name)
                        if frame_data:
                            frames_to_send[stream_name] = frame_data
                    
                    if frames_to_send:
                        for stream_name, frame_data in frames_to_send.items():
                            try:
                                self.socketio.emit('frame_update', {
                                    'stream': stream_name,
                                    'frame': frame_data
                                }, room=client_id)
                            except Exception as e:
                                print(f"Error sending frame for {stream_name}: {str(e)}")
                    
                    time.sleep(max(0.001, self.stream_manager.min_frame_time))
                        
                except Exception as e:
                    print(f"Error in frame push thread: {str(e)}")
                    time.sleep(0.1)
                    
        self.client_push_threads[client_id] = {
            'running': True,
            'thread': threading.Thread(
                target=push_frames,
                daemon=True
            )
        }
        self.client_push_threads[client_id]['thread'].start()

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
                metrics_dict = {}
                for key, value in self.latest_metrics.to_dict().items():
                    if hasattr(value, 'item'):
                        metrics_dict[key] = value.item()
                    else:
                        metrics_dict[key] = value
                return jsonify(metrics_dict)
        return jsonify({})

    def run(self):
        """Start the Flask-SocketIO server."""
        try:
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False,
                log_output=True,
                allow_unsafe_werkzeug=True
            )
        except KeyboardInterrupt:
            print("Shutting down web server...")
            for client_id in list(self.client_push_threads.keys()):
                self.client_push_threads[client_id]['running'] = False
            time.sleep(0.1)