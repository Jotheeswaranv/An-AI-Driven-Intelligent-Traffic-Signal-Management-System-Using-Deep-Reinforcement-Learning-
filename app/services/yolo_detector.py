"""
yolo_detector.py - YOLOv8 + TensorFlow hybrid detection engine.
"""
import cv2
import numpy as np
from app import logger
from app.services.tf_ambulance_detector import TFAmbulanceDetector

VEHICLE_CLASSES = {'car': 2, 'motorcycle': 3, 'bus': 5, 'truck': 7}
EMERGENCY_KEYWORDS = ['ambulance', 'fire truck', 'emergency', 'firetruck']
DENSITY_WEIGHTS = {
    'car': 1.0, 'motorcycle': 0.5, 'bus': 2.5,
    'truck': 2.0, 'ambulance': 1.0, 'default': 1.0,
}

_tf_detector = None
_tf_lock = None

def get_tf_detector():
    global _tf_detector, _tf_lock
    import threading
    if _tf_lock is None:
        _tf_lock = threading.Lock()
    with _tf_lock:
        if _tf_detector is None:
            logger.info('Initializing TensorFlow ambulance detector...')
            _tf_detector = TFAmbulanceDetector()
            logger.info('TensorFlow detector ready.')
    return _tf_detector


class YOLODetector:
    def __init__(self):
        self.model = None
        self.tf_detector = get_tf_detector()
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            import os
            model_file = 'best.pt' if os.path.exists('best.pt') else 'yolov8n.pt'
            self.model = YOLO(model_file)
            logger.info(f'YOLO model loaded: {model_file}')
        except Exception as e:
            logger.error(f'YOLO load failed: {e}')
            self.model = None

    def detect(self, frame, lane_key='lane1'):
        detections = []

        # Step 1: YOLO vehicle detection
        if self.model is not None:
            try:
                results = self.model(frame, verbose=False)[0]
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    label  = self.model.names[cls_id].lower()
                    conf   = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    is_vehicle = cls_id in VEHICLE_CLASSES.values() or any(
                        k in label for k in ['car','truck','bus','motorcycle','bike','van'])
                    is_emergency = (cls_id == 0 and 'ambulance' in self.model.names[0].lower()) \
                                   or any(kw in label for kw in EMERGENCY_KEYWORDS)
                    if is_vehicle or is_emergency:
                        detections.append({
                            'label': label, 'confidence': round(conf, 2),
                            'bbox': (x1, y1, x2, y2),
                            'is_emergency': is_emergency, 'method': 'YOLO',
                        })
            except Exception as e:
                logger.error(f'YOLO detect error: {e}')

        # Step 2: TensorFlow ambulance detection
        if not any(d['is_emergency'] for d in detections):
            try:
                tf_result = self.tf_detector.detect_ambulance(frame, lane_key)
                if tf_result['detected']:
                    bbox = tf_result.get('bbox') or (5, 5, 220, 90)
                    detections.append({
                        'label': 'ambulance',
                        'confidence': round(tf_result['confidence'], 2),
                        'bbox': bbox, 'is_emergency': True,
                        'method': tf_result.get('method', 'TF'),
                        'tf_score': tf_result.get('tf_score', 0),
                        'color_score': tf_result.get('color_score', 0),
                        'flash': tf_result.get('flash_detected', False),
                    })
                    logger.info(f'{lane_key}: Ambulance via {tf_result["method"]} '
                                f'conf={tf_result["confidence"]:.2f}')
            except Exception as e:
                logger.error(f'TF detection error: {e}')

        # Step 3: Mock vehicles if YOLO unavailable
        if self.model is None and len([d for d in detections if not d['is_emergency']]) < 2:
            detections.extend(self._mock_vehicles(frame))

        return detections

    def _mock_vehicles(self, frame):
        import random
        h, w = frame.shape[:2]
        labels = ['car', 'car', 'car', 'motorcycle', 'bus', 'truck']
        return [{
            'label': random.choice(labels),
            'confidence': round(random.uniform(0.60, 0.92), 2),
            'bbox': (random.randint(0,w-100), random.randint(0,h-80),
                     random.randint(60,w), random.randint(40,h)),
            'is_emergency': False, 'method': 'mock',
        } for _ in range(random.randint(3, 7))]

    def annotate_frame(self, frame, detections, signal='RED', green_time=0):
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            if det['is_emergency']:
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,0,255), 3)
                cv2.rectangle(annotated, (x1, max(0,y1-24)), (x2,y1), (0,0,200), -1)
                method = det.get('method','TF')
                cv2.putText(annotated, f"AMBULANCE {det['confidence']:.2f} [{method}]",
                            (x1+3, y1-6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (255,255,255), 1, cv2.LINE_AA)
                cv2.rectangle(annotated, (2,2),
                              (annotated.shape[1]-2, annotated.shape[0]-2),
                              (0,0,255), 4)
            else:
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,140,255), 2)
                cv2.putText(annotated, f"{det['label']} {det['confidence']:.2f}",
                            (x1, max(y1-5,12)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (0,200,255), 1, cv2.LINE_AA)

        # Signal overlay at bottom
        h = annotated.shape[0]
        sig_color = (0,220,0) if signal == 'GREEN' else (0,0,220)
        cv2.rectangle(annotated, (0, h-32), (annotated.shape[1], h), (0,0,0), -1)
        txt = f'SIGNAL: {signal}  |  Green: {green_time}s' if signal=='GREEN' else f'SIGNAL: {signal}'
        cv2.putText(annotated, txt, (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, sig_color, 2, cv2.LINE_AA)
        return annotated

    def compute_density(self, detections):
        score = 0.0
        for det in detections:
            weight = next((v for k,v in DENSITY_WEIGHTS.items()
                           if k in det['label']), DENSITY_WEIGHTS['default'])
            score += weight
        return round(score, 2)
