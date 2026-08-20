# Simple CamIO - Launch Guide

## How to Run the Application

The refactored code is **100% backward compatible** with the previous version. You can launch it exactly the same way as before.

### Method 1: Using Default Map (Recommended)

```powershell
python simple_camio.py
```

This will load the default map: `models/UkraineMap/UkraineMap.json`

### Method 2: Specifying a Custom Map

```powershell
python simple_camio.py --input1 models/CnapMap/CnapFirstFloor.json
```

Or the heart model:

```powershell
python simple_camio.py --input1 models/Heart/Heart.json
```

### Method 3: Unattended / Daemon

Pass `--camera` so startup never has to ask which camera to use, and
`--headless` to skip the preview window:

```bash
python simple_camio.py --headless --camera 0 --input1 models/CnapMap/CnapFirstFloor.json
```

## User Controls (Same as Before)

Once the application starts:

- **`q` or `ESC`**: Quit the application
- **`h`**: Manually trigger map re-detection (if tracking is lost)
- **`b`**: Toggle blip sounds on/off when moving between zones

Under `--headless` there is no window, so no keys are read. Stop the process
with Ctrl+C or `SIGTERM`; either one runs the full shutdown.

## What's Different (For Developers)

The launch command is unchanged, but the code was split out of the original
three flat modules into the `src/` package: `src/core/`, `src/detection/`,
`src/audio/`, `src/ui/` and `src/tap_classifier/`, with all tunable parameters
in `src/config.py`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the current layout and how the
threads fit together. It is the one place the structure is written down, so it
does not drift the way the copy that used to live here did.

## Troubleshooting

### "Module not found" errors

Modules inside `src/` import each other absolutely (`from src.config import ...`),
so two things matter:

- **Run from the repository root.** That is what puts `src` on the import path;
  from anywhere else `import src` fails.
- **Run submodules with `-m`, not as a file path.** `python src/tap_classifier/train_tap_classifier.py`
  raises `ModuleNotFoundError: No module named 'src'`; use
  `python -m src.tap_classifier.train_tap_classifier` instead.

### Camera not detected

Cameras are probed automatically. With exactly one working camera it is selected
silently. With several, you are prompted only when running in a terminal;
otherwise the first is used and a warning is logged, since a daemon has no one
to answer the prompt. Pass `--camera <port>` to skip detection altogether -
worth doing anyway, as one USB webcam often registers as two `/dev/video` nodes.

### Map not tracking

1. Press `h` to manually trigger re-detection
2. Ensure good lighting conditions
3. Check that the template image matches your physical map

## Configuration Changes

You can now easily adjust parameters without digging through code:

**Open `src/config.py`** and modify values in the configuration classes:

```python
# Example: Make tap detection more sensitive
class TapDetectionConfig:
    TAP_MIN_DURATION = 0.03  # Change from 0.05
    TAP_MAX_DURATION = 0.60  # Change from 0.50
```

**Enable debug logging:**

The logging level is set in `simple_camio.py`, near the top:

```python
logging.basicConfig(
    level=logging.DEBUG,  # change from logging.INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Requirements

Make sure you have all dependencies installed:

```powershell
pip install -r requirements.txt
```

`requirements.txt` is the authoritative list, and its comments explain why the
mediapipe range is pinned as narrowly as it is. Note that `mediapipe` above
0.10.21 will not work: those releases dropped the legacy Solutions API this code
is built on.

## Testing the Installation

Quick test to verify everything works:

```powershell
python -c "from src.detection import CombinedPoseDetector, SIFTModelDetectorMP; print('Import successful!')"
```

If you see "Import successful!" without errors, you're ready to run!

## Summary

✅ **Launch command is identical to before**
✅ **All functionality preserved**
✅ **Code is now modular and maintainable**
✅ **Configuration is centralized in `src/config.py`**
✅ **Better logging and error handling**

Just run:
```powershell
python simple_camio.py
```

And everything will work exactly as it did before, but with cleaner, more maintainable code!

