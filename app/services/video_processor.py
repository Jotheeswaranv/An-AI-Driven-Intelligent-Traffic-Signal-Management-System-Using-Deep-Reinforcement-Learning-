"""
video_processor.py
Parallel lane video processor with real-time signal updates.
"""
import cv2
import base64
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

from app.services.yolo_detector import YOLODetector
from app.services.traffic_logic import compute_signal_plan
from app import logger

SAMPLE_EVERY_N = 2        # sample every 2nd frame for speed
ROLLING_WINDOW = 6        # rolling average window


class LaneProcessor:
    def __init__(self, lane_key, video_path, detector, state, lock):
        self.lane_key        = lane_key
        self.video_path      = video_path
        self.detector        = detector
        self.state           = state
        self.lock            = lock
        self.density_history = deque(maxlen=ROLLING_WINDOW)

    def process(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f'{self.lane_key}: cannot open {self.video_path}')
            self._update(error='Cannot open video.', status='error')
            return

        frame_idx     = 0
        last_b64      = None
        peak_vehicles = 0
        has_emergency = False
        emergency_conf = 0.0
        emergency_method = ''

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % SAMPLE_EVERY_N != 0:
                    continue

                frame = cv2.resize(frame, (640, 360))
                detections = self.detector.detect(frame, lane_key=self.lane_key)

                # Emergency check
                emerg_dets = [d for d in detections if d['is_emergency']]
                if emerg_dets:
                    has_emergency   = True
                    emergency_conf  = max(d['confidence'] for d in emerg_dets)
                    emergency_method = emerg_dets[0].get('method', 'TF')
                    logger.info(
                        f'{self.lane_key}: EMERGENCY frame={frame_idx} '
                        f'conf={emergency_conf:.2f} method={emergency_method}'
                    )

                count   = len(detections)
                density = self.detector.compute_density(detections)
                self.density_history.append(density)
                peak_vehicles = max(peak_vehicles, count)

                rolling_density = round(
                    sum(self.density_history) / len(self.density_history), 1
                )

                # Get current signal for annotation
                with self.lock:
                    cur_signal     = self.state['results'].get(
                        self.lane_key, {}).get('signal', 'RED')
                    cur_green_time = self.state['results'].get(
                        self.lane_key, {}).get('green_time', 0)

                annotated = self.detector.annotate_frame(
                    frame, detections, cur_signal, cur_green_time
                )

                # Lane info overlay
                lane_num = self.lane_key.replace('lane', '')
                cv2.putText(annotated, f'Lane {lane_num}',
                            (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(annotated, f'Density: {rolling_density}',
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255,255,255), 1, cv2.LINE_AA)

                if has_emergency:
                    # Flashing red border
                    border_col = (0,0,255) if (frame_idx//3)%2==0 else (0,0,150)
                    cv2.rectangle(annotated, (0,0),
                                  (annotated.shape[1]-1, annotated.shape[0]-33),
                                  border_col, 5)
                    cv2.putText(annotated,
                                f'!!! EMERGENCY [{emergency_method}] !!!',
                                (120, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0,0,255), 2, cv2.LINE_AA)

                _, buf = cv2.imencode(
                    '.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 82]
                )
                last_b64 = base64.b64encode(buf).decode('utf-8')

                self._update(
                    vehicle_count   = count,
                    density         = rolling_density,
                    has_emergency   = has_emergency,
                    emergency_conf  = round(emergency_conf, 2),
                    emergency_method= emergency_method,
                    frame_b64       = last_b64,
                    status          = 'processing',
                )

        finally:
            cap.release()

        final_density = round(
            sum(self.density_history) / len(self.density_history), 1
        ) if self.density_history else 0

        self._update(
            vehicle_count   = peak_vehicles,
            density         = final_density,
            has_emergency   = has_emergency,
            emergency_conf  = round(emergency_conf, 2),
            emergency_method= emergency_method,
            frame_b64       = last_b64,
            status          = 'done',
        )
        logger.info(
            f'{self.lane_key}: done. density={final_density} '
            f'emergency={has_emergency}'
        )

    def _update(self, **kwargs):
        with self.lock:
            if self.lane_key not in self.state['results']:
                self.state['results'][self.lane_key] = {
                    'signal': 'RED', 'green_time': 0,
                    'has_emergency': False, 'density': 0,
                    'vehicle_count': 0, 'reason': '',
                    'frame_b64': None, 'status': 'waiting',
                    'emergency_conf': 0.0, 'emergency_method': '',
                }
            self.state['results'][self.lane_key].update(kwargs)
            _recompute_signals(self.state['results'])


def _recompute_signals(results: dict):
    plan = compute_signal_plan(results)
    for lane, sig in plan.items():
        if lane in results:
            results[lane]['signal']     = sig['signal']
            results[lane]['green_time'] = sig['green_time']
            results[lane]['reason']     = sig['reason']


class VideoProcessor:
    def __init__(self, lane_videos: dict, state: dict, lock):
        self.lane_videos = lane_videos
        self.state       = state
        self.lock        = lock
        self.detector    = YOLODetector()

    def process_all(self):
        processors = [
            LaneProcessor(key, path, self.detector, self.state, self.lock)
            for key, path in self.lane_videos.items()
        ]
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(p.process): p.lane_key
                       for p in processors}
            for future in as_completed(futures):
                lane = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f'{lane} failed: {e}')

        with self.lock:
            self.state['done']    = True
            self.state['running'] = False
        logger.info('All lanes processing complete.')
