# Simple CamIO - Refactored Architecture

## Overview

This document describes the refactored architecture of Simple CamIO, an interactive map system using hand tracking and gesture recognition.

## Project Structure

```
simple_camio/
├── config.py                  # Centralized configuration parameters
├── utils.py                   # Utility functions (camera, file I/O, drawing)
├── audio.py                   # Audio playback components
├── gesture_detection.py       # Movement filtering and gesture detection
├── pose_detector.py           # MediaPipe hand tracking and tap detection
├── sift_detector.py           # SIFT-based map tracking
├── interaction_policy.py      # Zone-based interaction logic
├── workers.py                 # Background worker threads
├── simple_camio.py            # Main application entry point
├── simple_camio_2d.py         # Legacy compatibility layer
├── simple_camio_mp.py         # Legacy compatibility layer
└── models/                    # Map configurations and assets
```

## Module Descriptions

### config.py
Centralized configuration module containing all tunable parameters:
- **CameraConfig**: Camera resolution, buffer size, processing scale
- **MovementFilterConfig**: Filter parameters for position smoothing
- **GestureDetectorConfig**: Thresholds for dwell/movement detection
- **TapDetectionConfig**: Parameters for single/double tap recognition
- **InteractionConfig**: Zone detection and touch thresholds
- **SIFTConfig**: Feature detection and matching parameters
- **MediaPipeConfig**: Hand tracking configuration
- **AudioConfig**: Volume levels and audio settings
- **UIConfig**: Display colors, fonts, and visual feedback
- **WorkerConfig**: Thread and queue parameters

### utils.py
Common utility functions:
- `list_camera_ports()`: Detect available cameras
- `select_camera_port()`: Auto-select or prompt for camera
- `load_map_parameters()`: Load JSON configuration files
- `draw_rectangle_on_image()`: Draw map tracking rectangle
- `draw_rectangle_from_points()`: Draw from pre-computed points
- `is_gesture_valid()`: Validate gesture data
- `normalize_gesture_location()`: Normalize gesture coordinates
- `color_to_index()`: Convert BGR colors to unique indices

### audio.py
Audio playback components:
- **AmbientSoundPlayer**: Looping background sounds (crickets, heartbeat)
  - `set_volume()`: Adjust playback volume
  - `play_sound()`: Start playback
  - `pause_sound()`: Pause playback
  
- **ZoneAudioPlayer**: Zone-based interactive audio
  - `play_welcome()`: Play welcome message
  - `play_goodbye()`: Play goodbye message
  - `convey()`: Play audio for zone interactions
  - `play_description()`: Play map description

### gesture_detection.py
Movement filtering and gesture analysis:
- **MovementFilter**: Exponential smoothing filter for positions
- **MovementMedianFilter**: Median filter for robust tracking
- **GestureDetector**: Detects dwell (still) vs moving gestures
  - Analyzes position history over time windows
  - Determines if user is dwelling at a location

### pose_detector.py
MediaPipe-based hand pose detection:
- **PoseDetectorMP**: Main hand tracking class
  - `detect()`: Process frame and detect hand poses
  - Recognizes pointing gestures (index finger extended)
  - Detects single and double taps using:
    - Z-depth changes (finger moving toward/away from camera)
    - Distal flexion angle (finger bending at tip joint)
  - Supports multi-hand tracking with stable hand identification
  - Adaptive thresholds based on noise levels

**Tap Detection Features:**
- Noise-adaptive thresholds for robustness
- Velocity-based press/release detection
- XY drift tolerance for natural finger movement
- Cooldown period to prevent repeated detections
- Combined Z-depth and angle analysis for reliability

### sift_detector.py
SIFT-based template detection and tracking:
- **SIFTModelDetectorMP**: Map template tracker
  - `detect()`: Detect template in camera frame
  - Uses SIFT features with ORB fallback
  - Corner feature enhancement for edges
  - MAGSAC++ for robust homography estimation
  - Periodic validation to detect tracking loss
  - Quality monitoring with automatic re-detection
  - `quick_validate_position()`: Fast position verification
  - `get_tracking_status()`: Status string for UI display

**Tracking Features:**
- Multi-stage detection with preprocessing (CLAHE, blur)
- Automatic re-detection on quality degradation
- Visual feedback with rectangle highlighting
- Position validation to detect map movement

### interaction_policy.py
Zone-based interaction logic:
- **InteractionPolicy2D**: Maps positions to zones
  - Color-coded zone detection
  - Mode filtering to reduce noise (ring buffer)
  - Z-threshold for touch detection
  - Handles out-of-bounds positions gracefully

### workers.py
Background processing threads:
- **PoseWorker**: Asynchronous hand pose detection
  - Processes frames in background
  - Thread-safe result sharing
  - Downscaled processing for speed
  
- **SIFTWorker**: Asynchronous template tracking
  - Full-resolution SIFT processing
  - Periodic validation scheduling
  - Multiple preprocessing attempts
  - Retry logic for robustness

**Benefits:**
- Non-blocking UI rendering
- Maintains high frame rates
- Parallel processing of heavy computations

### simple_camio.py
Main application entry point with structured functions:

**Initialization:**
- `initialize_system()`: Load config and create components
- `setup_camera()`: Configure camera capture
- `create_worker_threads()`: Start background workers
- `setup_signal_handler()`: Handle Ctrl+C gracefully

**Main Loop:**
- `run_main_loop()`: Main processing loop
- `feed_worker_queues()`: Push frames to workers
- `process_gestures_and_audio()`: Handle gesture events
- `draw_map_tracking()`: Render tracking rectangle
- `draw_ui_overlay()`: Display status information
- `handle_keyboard_input()`: Process user commands

**Cleanup:**
- `cleanup()`: Graceful shutdown and resource release

**User Controls:**
- `q` or `ESC`: Quit application
- `h`: Manually trigger map re-detection
- `b`: Toggle blip sounds on/off

## Design Principles

### 1. Separation of Concerns
Each module has a single, well-defined responsibility:
- Configuration is centralized
- Audio logic is isolated from detection logic
- UI rendering is separate from processing

### 2. Modularity
Components are loosely coupled and can be tested independently:
- Detectors can be swapped with minimal changes
- Filters can be configured or replaced
- Audio players are self-contained

### 3. Readability
Code is structured for understanding:
- Clear function names describing intent
- Comprehensive docstrings explaining behavior
- Logical grouping of related functionality
- Comments explaining complex algorithms

### 4. Maintainability
Easy to modify and extend:
- Configuration changes in one place
- Adding new gestures only requires detector changes
- New audio zones just need JSON updates
- Worker threads encapsulate async complexity

### 5. Error Handling
Robust error management:
- Graceful degradation when components fail
- Logging instead of print statements
- Try-except blocks for external resources
- Validation of user inputs and sensor data

## Configuration Guide

All tunable parameters are in `config.py`. Key settings:

### Camera Settings
```python
CameraConfig.DEFAULT_WIDTH = 1920
CameraConfig.DEFAULT_HEIGHT = 1080
CameraConfig.POSE_PROCESSING_SCALE = 0.35  # Lower = faster
```

### Gesture Detection
```python
GestureDetectorConfig.DWELL_TIME_THRESH = 0.75  # Seconds
GestureDetectorConfig.X_MVMNT_THRESH = 0.95     # Pixels
```

### Tap Detection
```python
TapDetectionConfig.TAP_MIN_DURATION = 0.05      # Seconds
TapDetectionConfig.TAP_MAX_DURATION = 0.50      # Seconds
TapDetectionConfig.TAP_MIN_INTERVAL = 0.05      # Between taps
TapDetectionConfig.TAP_MAX_INTERVAL = 1.00      # Between taps
```

### SIFT Tracking
```python
SIFTConfig.MIN_INLIER_COUNT = 10                # Minimum matches
SIFTConfig.REDETECT_INTERVAL = 150              # Frames
SIFTConfig.MIN_TRACKING_QUALITY = 8             # Quality threshold
```

## Logging

The application uses Python's logging module with configurable levels:

```python
# In config.py
LOG_LEVEL = logging.INFO  # Change to DEBUG for verbose output
```

Log messages include:
- Initialization steps
- Gesture detections (taps, movements)
- Tracking status changes
- Errors and warnings
- User actions

## Performance Considerations

### Threading Architecture
- **Main Thread**: Camera capture, UI rendering, event handling
- **Pose Worker**: Hand detection and gesture recognition
- **SIFT Worker**: Template detection and tracking

### Queue Management
- Queues have maxsize=1 to process only latest frames
- Old frames are dropped to prevent lag
- Non-blocking queue operations

### Processing Optimizations
- Pose detection uses downscaled images (35% of original)
- SIFT worker tries preprocessed variants first
- Validation is cheaper than full re-detection
- Periodic instead of per-frame validation

## Extending the System

### Adding New Gestures
1. Add detection logic in `pose_detector.py`
2. Update status strings in `run_main_loop()`
3. Add handling in `process_gestures_and_audio()`

### Adding New Audio Zones
1. Update map JSON with new hotspot
2. Add audio file to Audio/ directory
3. No code changes needed!

### Changing Detection Algorithm
1. Create new detector class following the interface
2. Update `initialize_system()` to use new detector
3. Ensure `detect()` method returns compatible format

### Adjusting Performance
1. Modify `POSE_PROCESSING_SCALE` for speed/accuracy trade-off
2. Adjust `REDETECT_INTERVAL` for tracking reliability
3. Change worker queue sizes in `WorkerConfig`

## Backward Compatibility

Legacy imports still work:
```python
from simple_camio_2d import InteractionPolicy2D, CamIOPlayer2D
from simple_camio_mp import PoseDetectorMP, SIFTModelDetectorMP
```

These modules now act as thin compatibility layers that import from the new modular structure.

## Troubleshooting

### Map Not Detected
- Check lighting conditions
- Ensure template image is clear and high-contrast
- Adjust `SIFTConfig.SIFT_CONTRAST_THRESHOLD`
- Try manual re-detection with `h` key

### Taps Not Recognized
- Verify hand is in pointing gesture
- Check `TapDetectionConfig` thresholds
- Enable DEBUG logging to see tap detection details
- Ensure finger movement is perpendicular to map

### Performance Issues
- Reduce `POSE_PROCESSING_SCALE`
- Increase `REDETECT_INTERVAL`
- Lower camera resolution
- Close other applications

## Future Enhancements

Potential improvements:
- [ ] Support for multiple simultaneous maps
- [ ] Gesture recording and playback
- [ ] Remote monitoring and control
- [ ] Custom gesture training
- [ ] Mobile device support
- [ ] Web-based configuration interface

