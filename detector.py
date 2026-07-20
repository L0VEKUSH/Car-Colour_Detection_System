"""
detector.py
 
YOLOv8 wrapper for the Car Colour Detection and Traffic Monitoring System,
plus the frame-annotation pipeline that ties object detection together
with colour classification.

Two things live here:

    VehicleDetector   - loads a YOLOv8 model and returns filtered
                         detections (car / person / bus / truck /
                         motorcycle only - everything else COCO can
                         detect is discarded).

    annotate_frame()  - a plain function that takes one frame plus a
                         VehicleDetector and a CarColorDetector, draws
                         all the required boxes/labels, and returns the
                         annotated frame + a stats dict + a details list.
                         It has no Tkinter dependency, so it can (and is)
                         unit-tested directly without a GUI.
"""

import time

import cv2

import utils

# COCO class indices this app cares about (from the pretrained YOLOv8
# checkpoint's default 80-class label set).
TARGET_CLASSES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class VehicleDetector:
    """Thin wrapper around an Ultralytics YOLOv8 model, filtered down to
    the traffic-relevant classes this app reports on."""

    def __init__(self, model_path, confidence=0.4):
        self.model_path = model_path
        self.confidence = confidence
        self.model = None
        self._load_error = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO  # imported lazily so utils.py
            # (and every module that imports it) stays free of the heavy
            # ultralytics/torch import even when detection isn't needed.
            self.model = YOLO(self.model_path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the GUI
            self.model = None
            self._load_error = str(exc)
            raise RuntimeError(
                f"Could not load YOLOv8 model from '{self.model_path}': {exc}"
            ) from exc

    @property
    def is_ready(self):
        return self.model is not None

    def set_confidence(self, value):
        self.confidence = max(0.05, min(0.95, float(value)))

    def detect(self, frame_bgr):
        """Run inference on one BGR frame. Returns a list of dicts:
        {class_id, class_name, confidence, bbox=(x1,y1,x2,y2)}, filtered
        to TARGET_CLASSES only. Returns [] if the model isn't loaded or
        nothing relevant was found - an empty result is a normal,
        expected outcome, not an error."""
        if self.model is None or frame_bgr is None or frame_bgr.size == 0:
            return []

        results = self.model.predict(
            source=frame_bgr, conf=self.confidence, verbose=False
        )
        if not results:
            return []

        detections = []
        r = results[0]
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in TARGET_CLASSES:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detections.append({
                "class_id": cls_id,
                "class_name": TARGET_CLASSES[cls_id],
                "confidence": float(box.conf[0]),
                "bbox": (x1, y1, x2, y2),
            })
        return detections


#        
# Drawing + orchestration
#        
_BOX_COLOR_BY_CLASS = {
    "bus": utils.BOX_COLOR_BUS,
    "truck": utils.BOX_COLOR_TRUCK,
    "motorcycle": utils.BOX_COLOR_MOTORCYCLE,
}


def draw_label(frame_bgr, bbox, color_bgr, text):
    """Draw a bounding box plus a filled label above it, with a text
    colour chosen for contrast against the box colour."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color_bgr, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, 0.55, 1)
    label_y0 = max(y1 - th - baseline - 6, 0)
    cv2.rectangle(frame_bgr, (x1, label_y0), (x1 + tw + 8, label_y0 + th + baseline + 6), color_bgr, -1)

    text_color = (255, 255, 255) if sum(color_bgr) < 400 else (0, 0, 0)
    cv2.putText(frame_bgr, text, (x1 + 4, label_y0 + th + 1), font, 0.55, text_color, 1, cv2.LINE_AA)


def annotate_frame(frame_bgr, vehicle_detector, color_detector):
    """Run detection + colour classification over one frame.

    Pure function - no Tkinter, no globals beyond the two detector
    objects passed in - so both the static "Detect" button and the
    live webcam/video loop can share exactly the same code path, and it
    can be unit-tested directly.

    Returns (annotated_frame_bgr, stats: dict, details: list[dict]).
    stats always contains: cars, blue_cars, other_cars, people,
    processing_time (seconds).
    """
    t0 = time.time()
    detections = vehicle_detector.detect(frame_bgr) if vehicle_detector else []
    annotated = frame_bgr.copy()
    h_frame, w_frame = frame_bgr.shape[:2]

    stats = {"cars": 0, "blue_cars": 0, "other_cars": 0, "people": 0}
    details = []

    for det in detections:
        cls = det["class_name"]
        x1, y1, x2, y2 = det["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        conf_pct = det["confidence"] * 100

        if cls == "car":
            crop = frame_bgr[y1:y2, x1:x2]
            color_name, purity = color_detector.detect_dominant_color(crop)
            is_blue = color_detector.is_blue(color_name)

            stats["cars"] += 1
            stats["blue_cars" if is_blue else "other_cars"] += 1

            box_color = utils.BOX_COLOR_BLUE_CAR if is_blue else utils.BOX_COLOR_OTHER_CAR
            label = f"{color_name.title()} ({conf_pct:.0f}%)"
            draw_label(annotated, (x1, y1, x2, y2), box_color, label)
            details.append({
                "class": "car", "color": color_name, "color_purity": purity,
                "yolo_confidence": round(det["confidence"] * 100, 2), "bbox": (x1, y1, x2, y2),
            })

        elif cls == "person":
            stats["people"] += 1
            draw_label(annotated, (x1, y1, x2, y2), utils.BOX_COLOR_PERSON, f"Person {conf_pct:.0f}%")
            details.append({"class": "person", "confidence": det["confidence"], "bbox": (x1, y1, x2, y2)})

        else:
            box_color = _BOX_COLOR_BY_CLASS.get(cls, utils.BOX_COLOR_DEFAULT)
            draw_label(annotated, (x1, y1, x2, y2), box_color, f"{cls.capitalize()} {conf_pct:.0f}%")
            details.append({"class": cls, "confidence": det["confidence"], "bbox": (x1, y1, x2, y2)})

    stats["processing_time"] = round(time.time() - t0, 3)
    return annotated, stats, details
