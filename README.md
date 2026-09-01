Simple CamIO 2D

Description: Simple CamIO 2D is a Python version of CamIO specialized to a flat, rectangular model such as a tactile map. This version relies on finger/hand tracking rather than the use of a stylus.

## Features

- **Hand Tracking**: MediaPipe-based hand pose detection with adaptive thresholds
- **Tap Detection**: Multi-modal tap recognition (Z-depth, angle, palm plane penetration)
- **Double-Tap Support**: Reliable double-tap detection for triggering actions
- **Spatial Audio**: Zone-based audio feedback with ambient soundscapes
- **SIFT Tracking**: Robust map tracking using SIFT/ORB feature matching
- **Threaded Architecture**: Non-blocking camera capture and display for high performance (400+ FPS)
- **Data Collection**: Automatic collection of tap detection data for classifier training
- **Headless Mode**: Run without display window - perfect for Raspberry Pi daemon deployment
- **Low-Power Sleep**: Drops to 1 FPS once the map is tracked and no hand has been seen for a while

## Data Collection and Classifier Training

Simple CamIO can now automatically collect tap detection data while you use the program. This allows you to train the tap classifier on your real-world usage patterns for improved accuracy!

**Quick Start:**

1. Run with collection enabled: `python simple_camio.py --collect-tap-data`
2. Perform taps as usual - data is collected automatically
3. Train on your data: `python -m src.tap_classifier.train_tap_classifier --train-from-collected --data-dir data/tap_dataset`

For detailed instructions, see [DATA_COLLECTION_GUIDE.md](src/tap_classifier/DATA_COLLECTION_GUIDE.md).

Requirements: To run Simple CamIO 2D, one needs to set up several things. 
- Firstly, There needs to be a json file that defines a model, that is it describes the components of an interactive map.  It contains the filenames of the various components of a model, as well as other important information such as the hotspot dictionary.  An example we recommend using as reference is `models/UkraineMap/UkraineMap.json`.

- Secondly, we require a printed map with text features along all four edges. An image of this map should be included, with its filename being specified in the element "template_image" of the model dictionary of the input json file.  We recommend using `models/UkraineMap/template.png` as an example to print out.

- Next, we require a digital version of the map that represents hotspot zones with unique indices as in `models/UkraineMap/UkraineMap.png`, and this filename should be specified in the element "filename" of the model dictionary of the input json file. Each index is a specific (R,G,B) color value. The image dimensions should match that of the template image. 

- Sound files, as named in the hotspots dictionary in the supplied json file, should be placed in the appropriate folder, as specified in the hotspots dictionary. The hotspots dictionary maps the zone index (from the zone map) to the sound file.

- Python 3.9+ installed with the required libraries specified in `requirements.txt`:
  - `mediapipe>=0.10.14,<0.10.22`
  - `numpy>=1.19.5,<1.27`
  - `scipy>=1.5.4,<2.0`
  - `opencv-contrib-python>=4.5.5.64,<5.0.0`
  - `pyglet>=1.5.0,<3.0.0`

For best performance, we recommend the camera sit above the map to get a fronto-parallel view as much as possible. The camera should have an unobstructed view of the 4 sides of the map, and the hand should be held such that the camera can clearly view it. The map should be close enough to the camera to take up most of the space in the camera image (so it is well resolved), but sufficient space (roughly 20 cm) between the map and the edges of the image should be available to ensure reliable finger/hand tracking even when the user is pointing to a feature near an edge of the map.

## Running Simple CamIO

To run with the default map (UkraineMap):
```powershell
python simple_camio.py
```

To run with a custom map:
```powershell
python simple_camio.py --input1 models/UkraineMap/UkraineMap.json
```

To pick a camera explicitly instead of auto-detecting one:
```bash
python simple_camio.py --camera 0
```
Auto-detection probes `/dev/video*` in order, which is slow and ambiguous: a
single USB webcam often registers as two nodes. Always pass `--camera` for
unattended runs.

To run in headless mode (no display window, suitable for Raspberry Pi daemon):
```bash
# Inside a virtual display, so pyglet can still reach an audio device
xvfb-run -a python simple_camio.py --headless --camera 0 --input1 models/CnapMap/CnapFirstFloor.json

# Or against an existing display
DISPLAY=:0 python simple_camio.py --headless --camera 0
```
Stop it with Ctrl+C or `SIGTERM`; both run the full shutdown (workers joined,
camera released, goodbye message played).

**Note for Raspberry Pi:** Headless mode requires `xvfb` or a `DISPLAY` environment variable for audio support.
Install with: `sudo apt-get install xvfb`

See [RASPBERRY_PI_DAEMON.md](RASPBERRY_PI_DAEMON.md) for detailed instructions on running as a Linux daemon/service.

### Keyboard Controls

While running:
- Press **`h`** to manually re-detect the map (reset homography)
- Press **`b`** to toggle zone transition blips on/off
- Press **`q`** or **`ESC`** to quit the application

## Usage Instructions

To use, simply make a pointing gesture by extending the index finger out and curling in the other fingers.  The area on the map indicated under the tip of the index finger will be dictated aloud.  The hand should be kept flat against the surface with the finger jutting out rather than the hand being held up above the map with the finger pointed down.
![](img/pointing_yes.jpg) ![](img/pointing_no.jpg)

## Installation

### Prerequisites
- Python 3.9 or higher
- A webcam or USB camera
- A printed tactile map with clear corner features

### Installation Steps

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/Coughlan-Lab/simple_camio.git
   cd simple_camio
   ```

2. **Create a virtual environment (recommended):**
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```powershell
   python -c "from src.detection import CombinedPoseDetector, SIFTModelDetectorMP; print('Installation successful!')"
   ```

## Project Structure

```
simple_camio/
├── simple_camio.py           # Main entry point (clean, UI-focused)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── ARCHITECTURE.md           # Detailed architecture documentation
├── LAUNCH_GUIDE.md          # Launch instructions
│
├── src/                     # Source code package
│   ├── __init__.py
│   ├── config.py           # Centralized configuration
│   │
│   ├── core/               # Core components
│   │   ├── utils.py        # Utility functions
│   │   ├── workers.py      # Background worker threads (Pose, SIFT, Audio)
│   │   ├── interaction_policy.py  # Zone mapping logic
│   │   ├── camera_thread.py       # Non-blocking camera capture
│   │   └── display_thread.py      # Non-blocking display rendering
│   │
│   ├── detection/          # Detection & tracking
│   │   ├── pose_detector.py      # Hand tracking and tap detection
│   │   ├── sift_detector.py      # SIFT-based map tracking
│   │   └── gesture_detection.py  # Movement filtering
│   │
│   ├── audio/              # Audio playback
│   │   └── audio.py        # Audio players
│   │
│   ├── ui/                 # User interface
│   │   └── display.py      # Drawing and overlays
│   │
│   └── tap_classifier/     # ML tap classification
│       ├── train_tap_classifier.py
│       ├── tap_classifier.py
│       ├── DATA_COLLECTION_GUIDE.md
│       └── TAP_CLASSIFIER_README.md
│
├── models/                  # Map configurations
│   ├── UkraineMap/         # Default map (central Kyiv)
│   ├── CnapMap/            # CNAP first floor plan
│   └── Heart/              # Anatomical heart model
│
├── data/
│   └── tap_dataset/        # Collected tap data for training
│
├── tests/                  # Unit tests (future)
```

## Advanced Features

### Tap Classifier Training

Train the tap classifier on synthetic data:
```powershell
python -m src.tap_classifier.train_tap_classifier --train --samples 1000
```

Train from your collected real-world data. This starts from the default weights
and replaces `models/tap_model.json`; add `--resume` to continue training the
model that is already there instead:
```powershell
python -m src.tap_classifier.train_tap_classifier --train-from-collected --data-dir data/tap_dataset
python -m src.tap_classifier.train_tap_classifier --train-from-collected --resume --data-dir data/tap_dataset
```

Evaluate the trained model:
```powershell
python -m src.tap_classifier.train_tap_classifier --evaluate
```

For more details, see [TAP_CLASSIFIER_README.md](src/tap_classifier/TAP_CLASSIFIER_README.md).

### Configuration

All tunable parameters are centralized in `src/config.py`:
- `CameraConfig` - Camera settings, processing scale, threaded capture/display options
- `TapDetectionConfig` - Tap detection thresholds and hand size scaling
- `SIFTConfig` - SIFT feature matching parameters
- `MediaPipeConfig` - MediaPipe hand tracking settings
- `AudioConfig` - Audio volume settings
- `UIConfig` - UI overlay settings
- `WorkerConfig` - Worker thread settings

**Performance Tuning:**
- Enable `USE_THREADED_CAPTURE=True` for non-blocking camera capture
- Enable `USE_THREADED_DISPLAY=True` for non-blocking display (recommended for high FPS)
- Adjust `DISPLAY_FRAME_SKIP` to control display rate (less critical with threaded display)

**Runtime overrides:**

A handful of settings can be changed at runtime instead of by editing
`src/config.py`, via a CLI flag or a `CAMIO_*` environment variable. A flag
wins over its environment variable, which wins over the class default:

| Flag | Environment variable | Overrides |
| --- | --- | --- |
| `--headless` | `CAMIO_HEADLESS=1` | `CameraConfig.HEADLESS` |
| `--resolution WxH` | `CAMIO_RESOLUTION=WxH` | `CameraConfig.DEFAULT_WIDTH`/`DEFAULT_HEIGHT` |
| `--camera-backend {any,dshow,msmf,v4l2}` | `CAMIO_CAMERA_BACKEND=...` | `CameraConfig.BACKEND` |
| `--collect-tap-data` | `CAMIO_COLLECT_TAP_DATA=1` | `TapDetectionConfig.COLLECT_TAP_DATA` |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | `CAMIO_LOG_LEVEL=...` | the root logger's level |

Run `python simple_camio.py --help` for the full flag list.

## Zone narration

Each hotspot's spoken description is generated, not recorded: `generate_audio`
reads a hotspot's `textDescription` (and, for the map-level clip, the model's
`mapDescriptionText`) and synthesizes a WAV with Piper, then rewrites the
model's `audioDescription` / `map_description` to point at it. Only missing
audio is produced - existing clips are left alone unless `--force` is passed.

```powershell
python -m src.tts.generate_audio --input1 models/UkraineMap/UkraineMap.json
```

This requires the `piper-tts` package and a downloaded voice model - install
with `uv pip install --python .venv/bin/python -r requirements-tts.txt`; it
is not in `requirements.txt` because it has no wheel for 32-bit Raspberry Pi
OS. See [docs/tts-setup.md](docs/tts-setup.md) for installing Piper and
fetching the `uk_UA-ukrainian_tts-medium` voice used by the bundled maps.

The generated WAVs are not committed to the repository - they are build
output from text that already lives in the model JSON. A map that ships
without its `tts/` audio is not broken: `ZoneAudioPlayer` synthesizes
whatever clips are missing the first time the model loads, so a fresh clone
or a map without pre-generated narration still speaks, at the cost of a
short delay on that first load.

## Troubleshooting

**Map not detected:**
- Ensure good lighting conditions
- Check that the template image matches your physical map
- Press `h` to manually trigger re-detection
- Adjust `SIFT_CONTRAST_THRESHOLD` in `SIFTConfig`

**Taps not detected:**
- Verify your pointing gesture (flat hand, extended index finger)
- Enable debug logging: `python simple_camio.py --log-level DEBUG`
- Check `scale_factor` values in logs (should be 0.35-1.0)
- Try collecting real-world data and retraining the classifier

**Performance issues:**
- Enable threaded capture: `USE_THREADED_CAPTURE=True` in `CameraConfig`
- Enable threaded display: `USE_THREADED_DISPLAY=True` in `CameraConfig`
- Lower `POSE_PROCESSING_SCALE` in `CameraConfig` (default: 0.35)
- Increase `REDETECT_INTERVAL` in `SIFTConfig` (default: 150 frames)
- Reduce camera resolution

**Expected Performance:**
- With threaded capture and display enabled: 60+ FPS processing
- Main loop should run smoothly without blocking on camera or display operations

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed system architecture
- [LAUNCH_GUIDE.md](LAUNCH_GUIDE.md) - Launch instructions and backward compatibility
- [.github/copilot-instructions.md](.github/copilot-instructions.md) - Developer guide for AI agents
- [src/tap_classifier/DATA_COLLECTION_GUIDE.md](src/tap_classifier/DATA_COLLECTION_GUIDE.md) - Data collection workflow
- [src/tap_classifier/TAP_CLASSIFIER_README.md](src/tap_classifier/TAP_CLASSIFIER_README.md) - Tap classifier details

__________________________________________________
## Legacy Installation Instructions

**Note:** The following instructions are for reference. Modern installation should follow the steps above.

### Old Method (Python 3.9.13)
1. Download and install Python 3.9.13 from https://www.python.org.
2. From the command line, in your working directory, type and enter "python -m venv camio"
3. Then type and enter "camio\Scripts\activate"
4. Then type and enter "git clone https://github.com/Coughlan-Lab/simple_camio.git"
5. Then type and enter "cd simple_camio" followed by "git fetch"
6. Then type and enter "pip install -r requirements.txt"

## Contributing

Contributions are welcome! Please see the developer documentation in `.github/copilot-instructions.md` for coding conventions and architecture details.

## License

See repository for license information.

## Citation

If you use Simple CamIO in your research, please cite the appropriate papers from the Coughlan Lab.
