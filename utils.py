
import os
import csv
import json
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk

                 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
ICONS_DIR = os.path.join(BASE_DIR, "icons")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

RECENT_FILES_PATH = os.path.join(ASSETS_DIR, "recent_files.json")
HISTORY_PATH = os.path.join(ASSETS_DIR, "detection_history.json")
CSV_REPORT_PATH = os.path.join(OUTPUTS_DIR, "detection_report.csv")

MAX_RECENT_FILES = 8
MAX_HISTORY_ENTRIES = 50


def ensure_app_dirs():
    """Create the application's working folders if they don't exist yet."""
    for d in (IMAGES_DIR, OUTPUTS_DIR, ICONS_DIR, ASSETS_DIR):
        os.makedirs(d, exist_ok=True)


def imread_safe(path):
    """Read an image from disk. Returns a BGR numpy array, or None if the
    path doesn't exist / isn't a readable image."""
    try:
        if not os.path.isfile(path):
            return None
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_safe(path, image_bgr):
    """Write a BGR numpy array to disk. Returns True on success."""
    try:
        ext = os.path.splitext(path)[1] or ".png"
        ok, buf = cv2.imencode(ext, image_bgr)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False



def resize_to_fit(image_bgr, max_w, max_h, allow_upscale=True):
    """Resize an image to fit within (max_w, max_h), preserving aspect
    ratio. Never distorts the image."""
    h, w = image_bgr.shape[:2]
    if w <= 0 or h <= 0 or max_w <= 0 or max_h <= 0:
        return image_bgr
    scale = min(max_w / w, max_h / h)
    if not allow_upscale:
        scale = min(scale, 1.0)
    if scale <= 0:
        return image_bgr
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=interp)


def bgr_to_tk_image(image_bgr):
    """Convert an OpenCV BGR numpy array into a Tkinter-displayable
    PhotoImage."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil_img)


CSV_FIELDNAMES = [
    "Image Name", "Timestamp", "Cars", "Blue Cars", "Other Cars",
    "People", "Processing Time (s)",
]


def append_csv_report(csv_path, row):
    """Append one detection-result row to a CSV report. Writes the header
    automatically the first time the file is created, so repeated saves
    build up a running report."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_recent_files():
    ensure_app_dirs()
    if not os.path.isfile(RECENT_FILES_PATH):
        return []
    try:
        with open(RECENT_FILES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [p for p in data if isinstance(p, str) and os.path.isfile(p)]
    except (json.JSONDecodeError, OSError):
        return []


def add_recent_file(path):
    files = load_recent_files()
    if path in files:
        files.remove(path)
    files.insert(0, path)
    files = files[:MAX_RECENT_FILES]
    try:
        ensure_app_dirs()
        with open(RECENT_FILES_PATH, "w", encoding="utf-8") as f:
            json.dump(files, f, indent=2)
    except OSError:
        pass
    return files

def load_history():
    ensure_app_dirs()
    if not os.path.isfile(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def add_history_entry(entry):
    history = load_history()
    entry = dict(entry)
    entry.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    history.insert(0, entry)
    history = history[:MAX_HISTORY_ENTRIES]
    try:
        ensure_app_dirs()
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except OSError:
        pass
    return history


DARK_THEME = {
    "bg": "#0f172a",
    "bg_alt": "#16213a",
    "card": "#182238",
    "toolbar": "#111c33",
    "fg": "#e6ebf5",
    "fg_muted": "#8fa0bf",
    "accent": "#f5a623",       
    "accent_fg": "#101826",
    "info": "#38bdf8",         
    "danger": "#ef4444",       
    "success": "#22c55e",      
    "border": "#28374f",
    "canvas_bg": "#060b16",
    "entry_bg": "#101a2e",
}

LIGHT_THEME = {
    "bg": "#f4f6fb",
    "bg_alt": "#e8edf6",
    "card": "#ffffff",
    "toolbar": "#eef2f9",
    "fg": "#101826",
    "fg_muted": "#5b6b85",
    "accent": "#c9791a",       
    "accent_fg": "#ffffff",
    "info": "#0284c7",
    "danger": "#dc2626",
    "success": "#16a34a",
    "border": "#d7dfec",
    "canvas_bg": "#e2e8f0",
    "entry_bg": "#ffffff",
}

BOX_COLOR_BLUE_CAR = (0, 0, 255)     
BOX_COLOR_OTHER_CAR = (255, 0, 0)   
BOX_COLOR_PERSON = (0, 200, 0)
BOX_COLOR_BUS = (255, 0, 220)
BOX_COLOR_TRUCK = (0, 140, 255)
BOX_COLOR_MOTORCYCLE = (255, 220, 0)
BOX_COLOR_DEFAULT = (180, 180, 180)

COLOR_NAME_TO_BGR = {
    "Blue": (255, 0, 0),
    "Red": (0, 0, 255),
    "White": (255, 255, 255),
    "Black": (35, 35, 35),
    "Silver": (198, 198, 198),
    "Gray": (140, 140, 140),
    "Yellow": (0, 255, 255),
    "Green": (0, 200, 0),
    "Orange": (0, 140, 255),
    "Brown": (19, 69, 139),
    "Unknown": (120, 120, 120),
}
