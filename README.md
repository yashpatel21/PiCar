# PiCar Autonomous Driving System

A real-time autonomous driving implementation for the PiCar platform using computer vision and geometric analysis for robust lane detection and vehicle control. The system employs adaptive thresholding and polynomial curve fitting to handle variable lighting conditions and complex track geometries.

## System Architecture

### Core Components
- Real-time image acquisition and processing pipeline
- Geometric lane detection with curve fitting algorithms  
- Adaptive steering control with dynamic speed adjustment
- Web-based monitoring and control interface
- Multi-threaded performance optimization

### Technical Stack
- Python 3.x
- OpenCV for image processing
- Flask/SocketIO for web interface
- NumPy for numerical computations
- PiCar SDK for hardware control
- Docker for containerized development and deployment
## Implementation Details

### Image Processing Pipeline
The system implements a sophisticated image processing pipeline utilizing adaptive thresholding techniques to isolate lane markers under varying lighting conditions. A Region of Interest (ROI) mask optimizes processing efficiency by focusing on relevant image areas. The pipeline employs sliding window detection combined with polynomial curve fitting to track lane lines across frames.

### Lane Detection Algorithm
Lane detection utilizes a two-phase approach:
1. Initial point detection through histogram analysis and sliding windows
2. Geometric validation using curve analysis and spatial relationships

The system handles both dual-lane and single-lane scenarios through sophisticated curve analysis, enabling robust performance in sharp turns and limited visibility conditions.

### Vehicle Control System
The control system implements:
- Dynamic steering angle calculation based on geometric lane analysis
- Exponential steering response for sharp turns
- Speed modulation based on curve severity
- Temporal smoothing for natural motion
- Safety-first approach with graceful degradation

### Real-time Visualization
Multiple visualization layers provide comprehensive system monitoring:
- Main view with lane detection overlay
- Binary threshold analysis 
- ROI visualization
- Point detection debugging
- Real-time metrics display

### Web Interface
Browser-based control interface featuring:
- Real-time video streaming with multiple debug views
- Dynamic parameter adjustment
- System status monitoring
- Performance metrics visualization
- Autonomous mode control

## Performance Characteristics

- Processing rate: 20-30 FPS on Raspberry Pi
- Sub-100ms latency for control decisions
- Adaptive to lighting variations
- Robust handling of sharp turns
- Graceful recovery from detection loss

## Safety Features

- Continuous lane detection validation
- Progressive speed reduction on detection loss
- Emergency stop capabilities
- Thread-safe state management
- Comprehensive system monitoring

## Technical Architecture

The system employs a multi-threaded architecture with careful consideration for:
- Thread synchronization in critical sections
- Non-blocking I/O for video streaming
- Efficient memory management
- Real-time performance optimization
- Robust error handling

## Practical Applications

While implemented on a miniature platform, the system demonstrates core autonomous driving principles applicable to full-scale vehicles:
- Real-time visual processing
- Dynamic path planning
- Predictive steering control
- Safety-first design philosophy
- Comprehensive monitoring systems

## Future Enhancements

Potential areas for lane following system enhancement:
- Integration of deep learning for improved lane detection
- Dynamic ROI adjustment
- Advanced path prediction algorithms
- Enhanced failure recovery mechanisms
- Performance optimization for resource-constrained environments

Enhancements for advanced autonomous driving system:
- Integration of deep learning for object detection (traffic signs, pedestrians, etc.)
- Object avoidance and emergency stop
- Advanced traffic logic and decision making based on object detection

Performance enhancements:
- Optimize processing rate and efficiency
- Increase FPS to 30+
- Reduce latency for control decisions

## Requirements

- Raspberry Pi (4B or newer)
- Raspberry Pi OS
- PiCar-X hardware platform
- Web browser for interface access

## Usage

Installation and setup instructions (WIP)

---
This implementation demonstrates production-grade autonomous driving principles while maintaining accessibility for educational purposes. The modular architecture allows for straightforward enhancement and adaptation to different platforms and requirements.