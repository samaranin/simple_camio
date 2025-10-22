import os
import sys
import cv2 as cv
import time
import numpy as np
import json
import argparse
import pyglet.media
from collections import deque
from simple_camio_2d import InteractionPolicy2D, CamIOPlayer2D
from simple_camio_mp import PoseDetectorMP, SIFTModelDetectorMP
import threading
import queue
import signal


class MovementFilter:
    def __init__(self):
        self.prev_position = None
        self.BETA = 0.5

    def push_position(self, position):
        if self.prev_position is None:
            self.prev_position = position
        else:
            self.prev_position = self.prev_position*(1-self.BETA) + position*self.BETA
        return self.prev_position


class MovementMedianFilter:
    def __init__(self):
        self.MAX_QUEUE_LENGTH = 30
        self.positions = deque(maxlen=self.MAX_QUEUE_LENGTH)
        self.times = deque(maxlen=self.MAX_QUEUE_LENGTH)
        self.AVERAGING_TIME = .7

    def push_position(self, position):
        self.positions.append(position)
        now = time.time()
        self.times.append(now)
        i = len(self.times)-1
        Xs = []
        Ys = []
        Zs = []
        while i >= 0 and now - self.times[i] < self.AVERAGING_TIME:
            Xs.append(self.positions[i][0])
            Ys.append(self.positions[i][1])
            Zs.append(self.positions[i][2])
            i -= 1
        return np.array([np.median(Xs), np.median(Ys), np.median(Zs)])

class GestureDetector:
    def __init__(self):
        self.MAX_QUEUE_LENGTH = 30
        self.positions = deque(maxlen=self.MAX_QUEUE_LENGTH)
        self.times = deque(maxlen=self.MAX_QUEUE_LENGTH)
        self.DWELL_TIME_THRESH = .75
        self.X_MVMNT_THRESH = 0.95
        self.Y_MVMNT_THRESH = 0.95
        self.Z_MVMNT_THRESH = 4.0

    def push_position(self, position):
        self.positions.append(position)
        now = time.time()
        self.times.append(now)
        i = len(self.times)-1
        Xs = []
        Ys = []
        Zs = []
        while (i >= 0 and now - self.times[i] < self.DWELL_TIME_THRESH):
            Xs.append(self.positions[i][0])
            Ys.append(self.positions[i][1])
            Zs.append(self.positions[i][2])
            i -= 1
        Xdiff = max(Xs) - min(Xs)
        Ydiff = max(Ys) - min(Ys)
        Zdiff = max(Zs) - min(Zs)
        print("(i: " + str(i) + ") X: " + str(Xdiff) + ", Y: " + str(Ydiff) + ", Z: " + str(Zdiff))
        if Xdiff < self.X_MVMNT_THRESH and Ydiff < self.Y_MVMNT_THRESH and Zdiff < self.Z_MVMNT_THRESH:
            return np.array([sum(Xs)/float(len(Xs)), sum(Ys)/float(len(Ys)), sum(Zs)/float(len(Zs))]), 'still'
        else:
            return position, 'moving'


class AmbientSoundPlayer:
    def __init__(self, soundfile):
        self.sound = pyglet.media.load(soundfile, streaming=False)
        self.player = pyglet.media.Player()
        self.player.queue(self.sound)
        self.player.eos_action = 'loop'
        self.player.loop = True

    def set_volume(self, volume):
        if 0 <= volume <= 1:
            self.player.volume = volume

    def play_sound(self):
        if not self.player.playing:
            self.player.play()

    def pause_sound(self):
        if self.player.playing:
            self.player.pause()


def draw_rect_in_image(image, sz, H):
    img_corners = np.array([[0,0],[sz[1],0],[sz[1],sz[0]],[0,sz[0]]], dtype=np.float32)
    img_corners = np.reshape(img_corners, [-1, 1, 2])
    H_inv = np.linalg.inv(H)
    pts = cv.perspectiveTransform(img_corners, H_inv)

    # Draw actual rectangle with lines instead of dots
    pts_int = np.int32(pts)
    cv.polylines(image, [pts_int], isClosed=True, color=(0, 255, 0), thickness=3)

    # Optionally draw corner dots for emphasis
    for pt in pts:
        cv.circle(image, (int(pt[0][0]), int(pt[0][1])), 5, (0, 255, 0), -1)

    return image


def draw_rect_pts(image, pts, color=(0,255,0), thickness=3):
    """Draw polygon from pts in same format as cv.perspectiveTransform output."""
    if pts is None:
        return image
    try:
        pts_int = np.int32(pts)
        cv.polylines(image, [pts_int], isClosed=True, color=color, thickness=thickness)
        # keep corner dots as subtle markers
        for pt in pts.reshape(-1, 2):
            cv.circle(image, (int(pt[0]), int(pt[1])), 4, color, -1)
    except Exception:
        pass
    return image

def select_cam_port():
    available_ports, working_ports, non_working_ports = list_ports()
    if len(working_ports) == 1:
        return working_ports[0][0]
    elif len(working_ports) > 1:
        print("The following cameras were detected:")
        for i in range(len(working_ports)):
            print(f'{i}) Port {working_ports[i][0]}: {working_ports[i][1]} x {working_ports[i][2]}')
        cam_selection = input("Please select which camera you would like to use: ")
        return working_ports[int(cam_selection)][0]
    else:
        return 0

def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 3:  # if there are more than 2 non working ports stop the testing.
        camera = cv.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." % dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" % (dev_port, h, w))
                working_ports.append((dev_port, h, w))
            else:
                print("Port %s for camera ( %s x %s) is present but does not read." % (dev_port, h, w))
                available_ports.append(dev_port)
        dev_port += 1
    return available_ports, working_ports, non_working_ports


# Function to load map parameters from a JSON file
def load_map_parameters(filename):
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            map_params = json.load(f)
            print("loaded map parameters from file.")
    else:
        print("No map parameters file found at " + filename)
        print("Usage: simple_camio.exe --input1 <filename>")
        print(" ")
        print("Press any key to exit.")
        _ = sys.stdin.read(1)
        exit(0)
    return map_params['model']


# Insert worker threads for async processing
class PoseWorker(threading.Thread):
    def __init__(self, pose_detector, in_queue, lock, processing_scale=0.5, stop_event=None):
        super().__init__(daemon=True)
        self.pose_detector = pose_detector
        self.in_queue = in_queue
        self.lock = lock
        self.processing_scale = processing_scale
        self.latest = (None, None, None)  # gesture_loc, status, annotated_image
        self.running = True
        self.stop_event = stop_event

    def run(self):
        while self.running and (self.stop_event is None or not self.stop_event.is_set()):
            try:
                frame, H = self.in_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            # run detector (downscaled inside)
            try:
                gesture_loc, gesture_status, annotated = self.pose_detector.detect(
                    frame, H, None, processing_scale=self.processing_scale, draw=True
                )
            except Exception as e:
                gesture_loc, gesture_status, annotated = None, None, None
            # Defensive normalization: ensure gesture_loc is either None or a 1D ndarray with 3 elements
            if gesture_loc is not None:
                arr = np.asarray(gesture_loc)
                if arr.size == 0:
                    gesture_loc = None
                elif arr.size >= 3:
                    if arr.size % 3 == 0 and arr.size > 3:
                        gesture_loc = arr.reshape(-1, 3)[-1].astype(float)
                    else:
                        gesture_loc = arr.flatten()[:3].astype(float)
                else:
                    gesture_loc = None

            with self.lock:
                self.latest = (gesture_loc, gesture_status, annotated)

    def stop(self):
        self.running = False
        if self.stop_event is not None:
            self.stop_event.set()


class SIFTWorker(threading.Thread):
    def __init__(self, sift_detector, in_queue, lock, stop_event=None):
        super().__init__(daemon=True)
        self.sift_detector = sift_detector
        self.in_queue = in_queue
        self.lock = lock
        self.running = True
        self.force_redetect = False
        self.validate_interval = 2.0
        self._last_validation_ts = 0.0
        self.stop_event = stop_event

    def run(self):
        """
        Process frames from the queue.
        - If homography is missing or forced: run full detection attempts.
        - Else: periodically run quick validation in the background.
        """
        RETRIES = 3
        while self.running and (self.stop_event is None or not self.stop_event.is_set()):
            try:
                frame = self.in_queue.get(timeout=0.2)  # frame is expected to be grayscale full-res
            except queue.Empty:
                continue

            try:
                # If we already have a homography and no force, do lightweight validation on a cadence
                if (not self.sift_detector.requires_homography) and (not self.force_redetect):
                    now = time.time()
                    if now - self._last_validation_ts >= self.validate_interval:
                        self._last_validation_ts = now
                        valid = self.sift_detector.quick_validate_position(frame, min_matches=6, position_threshold=40)
                        if not valid:
                            # Mark stale and immediately try a re-detect using the current frame
                            self.sift_detector.requires_homography = True
                            self.sift_detector.last_rect_pts = None
                            # continue to detection path below
                        else:
                            # Nothing else to do this iteration
                            continue

                # Either we need homography or a manual re-detect was requested
                detected = False
                attempts = [frame]
                # CLAHE
                try:
                    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    attempts.append(clahe.apply(frame))
                except Exception:
                    pass
                # slight blur
                try:
                    attempts.append(cv.GaussianBlur(frame, (5, 5), 0))
                except Exception:
                    pass

                for attempt_img in attempts:
                    try:
                        retval, H, _ = self.sift_detector.detect(attempt_img, force_redetect=self.force_redetect)
                    except Exception:
                        retval, H = False, None

                    # consume force flag
                    if self.force_redetect:
                        self.force_redetect = False

                    if retval and H is not None:
                        detected = True
                        break

                if not detected:
                    for _ in range(RETRIES):
                        try:
                            retval, H, _ = self.sift_detector.detect(frame, force_redetect=False)
                        except Exception:
                            retval, H = False, None
                        if retval and H is not None:
                            break

            except Exception as e:
                print(f"SIFTWorker error: {e}")
                # loop continues

    def trigger_redetect(self):
        """Manually trigger re-detection"""
        self.force_redetect = True

    def stop(self):
        self.running = False
        if self.stop_event is not None:
            self.stop_event.set()


parser = argparse.ArgumentParser(description='Code for CamIO.')
parser.add_argument('--input1', help='Path to parameter json file.', default='models/UkraineMap/UkraineMap.json')
args = parser.parse_args()

model = load_map_parameters(args.input1)

cam_port = select_cam_port()

# Initialize objects
if model["modelType"] == "sift_2d_mediapipe":
    model_detector = SIFTModelDetectorMP(model)
    pose_detector = PoseDetectorMP(model)
    gesture_detector = GestureDetector()
    motion_filter = MovementMedianFilter()
    interact = InteractionPolicy2D(model)
    camio_player = CamIOPlayer2D(model)
    camio_player.play_welcome()
    crickets_player = AmbientSoundPlayer(model['crickets'])
    heartbeat_player = AmbientSoundPlayer(model['heartbeat'])

heartbeat_player.set_volume(.05)
cap = cv.VideoCapture(cam_port, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FOCUS, 0)
# Reduce latency in some backends
try:
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
except Exception:
    pass

# Create queues and workers
pose_queue = queue.Queue(maxsize=1)
sift_queue = queue.Queue(maxsize=1)
lock = threading.Lock()

# Add a stop event and attach a SIGINT handler
stop_event = threading.Event()
def _signal_handler(sig, frame):
    print("Signal received, shutting down...")
    stop_event.set()
signal.signal(signal.SIGINT, _signal_handler)

pose_worker = PoseWorker(pose_detector, pose_queue, lock, processing_scale=0.35, stop_event=stop_event)
sift_worker = SIFTWorker(model_detector, sift_queue, lock, stop_event=stop_event)
pose_worker.start()
sift_worker.start()

timer = time.time() - 1
print("Controls: 'h'=re-detect map, 'b'=toggle blips, 'q'=quit")

# Add small helper near main loop to check gesture validity
def _gesture_valid(g):
    return (g is not None) and (hasattr(g, "__len__")) and (np.asarray(g).size >= 3)

# Remove validate_counter; keep flash counters
rect_flash_remaining = 0
RECT_FLASH_FRAMES = 10

# Main loop - avoid heavy work in the UI thread
try:
    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("No camera image returned.")
            break

        now_time = time.time()

        # Prepare grayscale once and always feed the SIFT worker the newest frame (queue drops old)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        try:
            sift_queue.put_nowait(gray)
        except queue.Full:
            try:
                _ = sift_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                sift_queue.put_nowait(gray)
            except queue.Full:
                pass

        # Push frame for pose worker (no copy; detect() will not draw by default)
        H_current = model_detector.H if model_detector.H is not None else np.eye(3)
        try:
            pose_queue.put_nowait((frame, H_current))
        except queue.Full:
            try:
                _ = pose_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                pose_queue.put_nowait((frame, H_current))
            except queue.Full:
                pass

        # Retrieve latest pose worker output
        with lock:
            gesture_loc, gesture_status, annotated = pose_worker.latest

        display_img = frame if (annotated is None) else annotated

        # If detector reports homography was updated, trigger flash highlight
        if getattr(model_detector, 'homography_updated', False):
            rect_flash_remaining = RECT_FLASH_FRAMES
            model_detector.homography_updated = False

        # PROCESS GESTURES AND INTERACTION
        if model_detector.H is None:
            heartbeat_player.pause_sound()
            crickets_player.play_sound()
        else:
            # Ensure age counter increments even if detect() isn't called every frame
            try:
                model_detector.frames_since_last_detection += 1
            except Exception:
                model_detector.frames_since_last_detection = 1

            # Draw rectangle from stored projected pts if available
            if getattr(model_detector, 'last_rect_pts', None) is not None:
                if rect_flash_remaining > 0:
                    display_img = draw_rect_pts(display_img, model_detector.last_rect_pts, color=(0, 255, 255), thickness=5)
                    rect_flash_remaining -= 1
                else:
                    display_img = draw_rect_pts(display_img, model_detector.last_rect_pts, color=(0, 255, 0), thickness=3)
            else:
                display_img = draw_rect_in_image(display_img, interact.image_map_color.shape, model_detector.H)

            if not _gesture_valid(gesture_loc):
                heartbeat_player.pause_sound()
            else:
                heartbeat_player.play_sound()
                zone_id = interact.push_gesture(gesture_loc)
                camio_player.convey(zone_id, gesture_status)

        # Add status overlay
        status_text = model_detector.get_tracking_status()
        cv.putText(display_img, status_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Add FPS
        prev_time = timer
        timer = time.time()
        elapsed_time = timer - prev_time
        if elapsed_time > 0:
            fps_text = f"FPS: {1/elapsed_time:.1f}"
            cv.putText(display_img, fps_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv.imshow('image reprojection', display_img)
        waitkey = cv.waitKey(1)
        if waitkey == 27 or waitkey == ord('q'):
            print('Exiting...')
            stop_event.set()
            break
        if waitkey == ord('h'):
            print("Manual re-detection triggered by user")
            gray_now = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            model_detector.requires_homography = True
            model_detector.last_rect_pts = None
            for _ in range(3):
                try:
                    sift_queue.put_nowait(gray_now)
                except queue.Full:
                    try:
                        _ = sift_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        sift_queue.put_nowait(gray_now)
                    except queue.Full:
                        pass
            sift_worker.trigger_redetect()
        if waitkey == ord('b'):
            camio_player.enable_blips = not camio_player.enable_blips
            print("Blips " + ("enabled" if camio_player.enable_blips else "disabled"))

        pyglet.clock.tick()
        pyglet.app.platform_event_loop.dispatch_posted_events()

except KeyboardInterrupt:
    print("KeyboardInterrupt received, shutting down...")
    stop_event.set()

# shutdown
pose_worker.stop()
sift_worker.stop()
# give threads a moment to exit cleanly
pose_worker.join(timeout=2.0)
sift_worker.join(timeout=2.0)

try:
    camio_player.play_goodbye()
except Exception:
    pass
heartbeat_player.pause_sound()
crickets_player.pause_sound()
time.sleep(1)
cap.release()
cv.destroyAllWindows()
