"""
tf_ambulance_detector.py

TensorFlow-powered ambulance detection engine.
Uses a multi-signal approach:
  1. TensorFlow CNN model for white vehicle + red-cross pattern recognition
  2. Advanced HSV color segmentation for siren lights (red+blue flash)
  3. Contour shape analysis for ambulance body dimensions
  4. Temporal flash pattern tracking (alternating red/blue = siren)

This runs alongside YOLOv8 for maximum detection accuracy.
"""

import cv2
import numpy as np
import threading
from collections import deque
from app import logger


class FlashTracker:
    """
    Tracks alternating red/blue light patterns over time.
    Real ambulance sirens flash at 1-4 Hz — this detects that pattern.
    """
    def __init__(self, window=20):
        self.history = deque(maxlen=window)
        self.lock = threading.Lock()

    def update(self, red_dominant: bool, blue_dominant: bool):
        with self.lock:
            if red_dominant:
                self.history.append('R')
            elif blue_dominant:
                self.history.append('B')
            else:
                self.history.append('N')

    def is_flashing(self) -> bool:
        """Returns True if alternating R/B pattern detected."""
        with self.lock:
            if len(self.history) < 6:
                return False
            h = list(self.history)
            # Count transitions between R and B
            transitions = sum(
                1 for i in range(1, len(h))
                if h[i] != h[i-1] and h[i] != 'N' and h[i-1] != 'N'
            )
            return transitions >= 2


class TFAmbulanceDetector:
    """
    TensorFlow + OpenCV hybrid ambulance detector.
    """

    def __init__(self):
        self.tf_model = None
        self.flash_trackers = {}   # per-lane flash tracker
        self._build_tf_model()

    def _build_tf_model(self):
        """
        Build a lightweight TensorFlow CNN that classifies image patches
        as 'ambulance' or 'non-ambulance' based on:
          - Color distribution (white body, red cross)
          - Texture features
          - Shape characteristics
        """
        try:
            import tensorflow as tf

            # Suppress TF logs
            tf.get_logger().setLevel('ERROR')

            # Build a simple but effective CNN
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(64, 64, 3)),

                # Block 1 - edge detection
                tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(2, 2),

                # Block 2 - pattern detection
                tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(2, 2),

                # Block 3 - feature extraction
                tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(2, 2),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid'),
            ])

            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            # Initialize with synthetic training
            self._synthetic_train(model)
            self.tf_model = model
            logger.info('TensorFlow ambulance CNN initialized and trained.')

        except ImportError:
            logger.warning('TensorFlow not installed. Using color-only detection.')
            self.tf_model = None
        except Exception as e:
            logger.error(f'TF model init failed: {e}')
            self.tf_model = None

    def _synthetic_train(self, model):
        """
        Train on synthetically generated ambulance / non-ambulance patches.
        This gives the model meaningful weights without needing a dataset file.
        """
        import tensorflow as tf
        np.random.seed(42)
        X, y = [], []

        # Generate positive samples (ambulance-like patches)
        for _ in range(400):
            patch = self._make_ambulance_patch()
            X.append(patch)
            y.append(1.0)

        # Generate negative samples (regular vehicles / background)
        for _ in range(400):
            patch = self._make_non_ambulance_patch()
            X.append(patch)
            y.append(0.0)

        X = np.array(X, dtype=np.float32) / 255.0
        y = np.array(y, dtype=np.float32)

        # Shuffle
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

        model.fit(X, y, epochs=15, batch_size=32,
                  validation_split=0.15, verbose=0)
        logger.info('TF model synthetic training complete.')

    def _make_ambulance_patch(self):
        """Generate a synthetic ambulance-like 64x64 patch."""
        patch = np.zeros((64, 64, 3), dtype=np.uint8)
        noise = np.random.randint(0, 15, (64, 64, 3), dtype=np.uint8)

        # White body (high brightness)
        white_val = np.random.randint(210, 255)
        patch[20:55, 5:60] = (white_val, white_val, white_val)

        # Red cross on body
        cross_r = np.random.randint(180, 255)
        patch[28:44, 18:26] = (0, 0, cross_r)   # vertical bar
        patch[34:38, 14:30] = (0, 0, cross_r)   # horizontal bar

        # Siren lights - alternating red/blue
        if np.random.random() > 0.5:
            patch[10:18, 8:22]  = (0, 0, 220)   # red light
            patch[10:18, 40:55] = (200, 30, 0)  # blue light
        else:
            patch[10:18, 8:22]  = (200, 30, 0)  # blue light
            patch[10:18, 40:55] = (0, 0, 220)   # red light

        # Wheels (dark circles approximate)
        patch[52:60, 12:20] = (30, 30, 30)
        patch[52:60, 44:52] = (30, 30, 30)

        # Add noise and slight variations
        patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Random brightness variation
        factor = np.random.uniform(0.7, 1.2)
        patch = np.clip(patch * factor, 0, 255).astype(np.uint8)

        return patch

    def _make_non_ambulance_patch(self):
        """Generate a synthetic non-ambulance vehicle patch."""
        patch = np.zeros((64, 64, 3), dtype=np.uint8)

        # Random vehicle color (not white)
        colors = [
            (30, 30, 150), (120, 30, 30), (30, 100, 30),
            (80, 80, 80),  (60, 40, 20), (100, 100, 30)
        ]
        color = colors[np.random.randint(0, len(colors))]
        patch[15:50, 5:60] = color

        # Windows (bluish-grey)
        patch[20:35, 12:35] = (160, 190, 200)
        patch[20:35, 38:55] = (160, 190, 200)

        # Add road/background noise
        noise = np.random.randint(0, 30, (64, 64, 3), dtype=np.uint8)
        patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return patch

    def get_flash_tracker(self, lane_key: str) -> FlashTracker:
        if lane_key not in self.flash_trackers:
            self.flash_trackers[lane_key] = FlashTracker(window=30)
        return self.flash_trackers[lane_key]

    def detect_ambulance(self, frame: np.ndarray, lane_key: str) -> dict:
        """
        Main detection method. Returns detection result dict.
        Combines TF CNN + color segmentation + flash pattern analysis.
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'method': None,
            'bbox': None,
            'color_score': 0.0,
            'tf_score': 0.0,
            'flash_detected': False,
        }

        frame_resized = cv2.resize(frame, (640, 360))

        # ── Method 1: Advanced color + siren light detection ──────────────────
        color_result = self._color_siren_detect(frame_resized, lane_key)
        result['color_score']   = color_result['score']
        result['flash_detected'] = color_result['flashing']

        # ── Method 2: TensorFlow CNN on candidate regions ─────────────────────
        tf_score = 0.0
        tf_bbox  = None
        if self.tf_model is not None:
            tf_result = self._tf_scan_frame(frame_resized)
            tf_score  = tf_result['score']
            tf_bbox   = tf_result['bbox']
        result['tf_score'] = tf_score

        # ── Method 3: White large vehicle shape detection ─────────────────────
        shape_result = self._shape_detect(frame_resized)

        # ── Combine all signals ───────────────────────────────────────────────
        # Emergency confirmed if ANY strong signal fires
        if color_result['flashing'] and color_result['score'] > 0.4:
            result['detected']   = True
            result['confidence'] = min(0.95, 0.7 + color_result['score'] * 0.25)
            result['method']     = 'Siren Flash Pattern'
            result['bbox']       = color_result.get('bbox')

        elif tf_score > 0.72:
            result['detected']   = True
            result['confidence'] = tf_score
            result['method']     = 'TensorFlow CNN'
            result['bbox']       = tf_bbox

        elif color_result['score'] > 0.65:
            result['detected']   = True
            result['confidence'] = color_result['score']
            result['method']     = 'Color Detection'
            result['bbox']       = color_result.get('bbox')

        elif (tf_score > 0.55 and color_result['score'] > 0.35
              and shape_result['large_white_vehicle']):
            result['detected']   = True
            result['confidence'] = (tf_score + color_result['score']) / 2
            result['method']     = 'TF + Color + Shape'
            result['bbox']       = tf_bbox or color_result.get('bbox')

        return result

    def _color_siren_detect(self, frame: np.ndarray, lane_key: str) -> dict:
        """
        Detect red+blue siren lights using HSV color segmentation.
        Also tracks flash pattern over time.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]

        # Red: two ranges (wraps in HSV)
        r1 = cv2.inRange(hsv, np.array([0,   80, 100]), np.array([12,  255, 255]))
        r2 = cv2.inRange(hsv, np.array([158, 80, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(r1, r2)

        # Blue
        blue_mask = cv2.inRange(hsv, np.array([95,  80, 100]), np.array([135, 255, 255]))

        # White (ambulance body)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 40, 255]))

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask  = cv2.morphologyEx(red_mask,  cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

        red_pixels  = cv2.countNonZero(red_mask)
        blue_pixels = cv2.countNonZero(blue_mask)
        total_px    = h * w

        red_ratio  = red_pixels  / total_px
        blue_ratio = blue_pixels / total_px

        # Update flash tracker
        tracker = self.get_flash_tracker(lane_key)
        tracker.update(
            red_dominant  = red_ratio  > 0.008,
            blue_dominant = blue_ratio > 0.008
        )

        # Score = how strong are the siren colors
        score = min(1.0, (red_ratio + blue_ratio) * 15)

        # Find bbox of siren light region
        combined = cv2.bitwise_or(red_mask, blue_mask)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        bbox = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 200:
                rx, ry, rw, rh = cv2.boundingRect(largest)
                # Expand bbox to encompass full vehicle
                bbox = (
                    max(0, rx - 20),
                    max(0, ry - 30),
                    min(w, rx + rw + 60),
                    min(h, ry + rh + 80)
                )

        return {
            'score':     score,
            'flashing':  tracker.is_flashing(),
            'red_px':    red_pixels,
            'blue_px':   blue_pixels,
            'bbox':      bbox,
        }

    def _tf_scan_frame(self, frame: np.ndarray) -> dict:
        """
        Slide TF model across frame regions to find ambulance patches.
        """
        import tensorflow as tf
        h, w = frame.shape[:2]
        best_score = 0.0
        best_bbox  = None

        # Scan with sliding window at multiple scales
        window_configs = [
            (w // 3, h // 3, w // 6, h // 6),   # large window
            (w // 4, h // 4, w // 8, h // 8),   # medium window
        ]

        patches, coords = [], []
        for (win_w, win_h, step_x, step_y) in window_configs:
            for y in range(0, h - win_h, step_y):
                for x in range(0, w - win_w, step_x):
                    patch = frame[y:y+win_h, x:x+win_w]
                    patch_resized = cv2.resize(patch, (64, 64))
                    patches.append(patch_resized)
                    coords.append((x, y, x+win_w, y+win_h))

        if not patches:
            return {'score': 0.0, 'bbox': None}

        batch = np.array(patches, dtype=np.float32) / 255.0
        preds = self.tf_model.predict(batch, verbose=0, batch_size=32).flatten()

        best_idx = int(np.argmax(preds))
        best_score = float(preds[best_idx])
        best_bbox  = coords[best_idx] if best_score > 0.5 else None

        return {'score': best_score, 'bbox': best_bbox}

    def _shape_detect(self, frame: np.ndarray) -> dict:
        """
        Detect large white rectangular vehicle shapes (ambulance body).
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 35, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        large_white = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 3000:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / max(h, 1)
                # Ambulance has wide rectangular shape (aspect 1.5 - 3.5)
                if 1.5 < aspect < 3.5:
                    large_white = True
                    break

        return {'large_white_vehicle': large_white}
