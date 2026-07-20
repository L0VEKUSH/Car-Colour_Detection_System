

import os
import time
import queue
import threading
import collections
from datetime import datetime

import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import utils
import detector
from detector import VehicleDetector
from color_detector import CarColorDetector


class ScrollableFrame(ttk.Frame):
    """A vertically scrollable container. Mouse-wheel scrolling is only
    bound while the cursor is over this widget, so it doesn't hijack
    scroll events elsewhere in the window."""

    def __init__(self, parent, bg, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=bg)

        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self._window, width=e.width))

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def set_bg(self, bg):
        self.canvas.configure(bg=bg)
        self.inner.configure(bg=bg)


class CarColorDetectionApp:
    APP_TITLE = "Car Colour Detection and Traffic Monitoring System"

    def __init__(self, root):
        self.root = root
        self.root.title(self.APP_TITLE)

        #   state  
        self.dark_mode = True
        self.theme = utils.DARK_THEME
        self.confidence = 0.6

        self.current_image_bgr = None    
        self.display_image_bgr = None      
        self.image_path = None
        self.zoom_factor = 1.0

        self.webcam_running = False
        self.video_running = False
        self.cap = None
        self._capture_thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        self._fps_history = collections.deque(maxlen=15)

        self._themed_widgets = []          
        self.stats_vars = {}

        #   models  
        self.status_var = tk.StringVar(value="Starting…")
        self.color_detector = CarColorDetector()
        self.detector = None
        try:
            self.detector = VehicleDetector(utils.MODEL_PATH, confidence=self.confidence)
        except Exception as exc:
            messagebox.showerror(
                "Model Error",
                "Could not load the YOLOv8 model (yolov8n.pt).\n\n"
                f"{exc}\n\n"
                "Detection features will stay disabled until a valid "
                "yolov8n.pt is available next to main.py.",
            )

        utils.ensure_app_dirs()
        self._build_ui()
        self._refresh_recent_files_menu()
        self._refresh_history_list()
        self._apply_theme()

        self.status_var.set("Ready" if (self.detector and self.detector.is_ready) else "Model unavailable — detection disabled")

                  
    def _build_ui(self):
        self.root.configure(bg=self.theme["bg"])

        self.menu_bar = tk.Menu(self.root)
        self._build_menu()
        self.root.config(menu=self.menu_bar)

        self.outer = tk.Frame(self.root, bg=self.theme["bg"])
        self.outer.pack(fill="both", expand=True)
        self._themed_widgets.append((self.outer, "bg"))

        self._build_header(self.outer)
        self._build_toolbar(self.outer)

        self.body = tk.Frame(self.outer, bg=self.theme["bg"])
        self.body.pack(fill="both", expand=True, padx=12, pady=(6, 6))
        self._themed_widgets.append((self.body, "bg"))

        self._build_preview_area(self.body)
        self._build_sidebar(self.body)

        self._build_statusbar(self.outer)

    def _build_menu(self):
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Open Image…", command=self.on_upload_image)
        file_menu.add_command(label="Open Video…", command=self.on_load_video)
        self.recent_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent Files", menu=self.recent_menu)
        file_menu.add_separator()
        file_menu.add_command(label="Save Result…", command=self.on_save_result)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_exit)
        self.menu_bar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(self.menu_bar, tearoff=0)
        view_menu.add_command(label="Toggle Dark / Light Mode", command=self.on_toggle_theme)
        view_menu.add_command(label="Zoom In", command=self.on_zoom_in)
        view_menu.add_command(label="Zoom Out", command=self.on_zoom_out)
        view_menu.add_command(label="Reset Zoom", command=self.on_zoom_reset)
        view_menu.add_command(label="Fullscreen Preview", command=self.on_fullscreen_preview)
        self.menu_bar.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)

    def _build_header(self, parent):
        header = tk.Frame(parent, bg=self.theme["toolbar"], height=64)
        header.pack(fill="x")
        self._themed_widgets.append((header, "toolbar"))

        title_box = tk.Frame(header, bg=self.theme["toolbar"])
        title_box.pack(side="left", padx=16, pady=8)
        self._themed_widgets.append((title_box, "toolbar"))

        title = tk.Label(title_box, text="🚦 Car Colour Detection & Traffic Monitoring",
                          font=("Segoe UI", 16, "bold"), bg=self.theme["toolbar"], fg=self.theme["fg"])
        title.pack(anchor="w")
        self._themed_widgets.append((title, "toolbar_fg"))

        subtitle = tk.Label(title_box, text="YOLOv8-powered vehicle colour & pedestrian analysis",
                             font=("Segoe UI", 9), bg=self.theme["toolbar"], fg=self.theme["fg_muted"])
        subtitle.pack(anchor="w")
        self._themed_widgets.append((subtitle, "toolbar_fg_muted"))

        self.theme_btn = tk.Button(header, text="☀ Light Mode", command=self.on_toggle_theme,
                                    relief="flat", cursor="hand2", font=("Segoe UI", 9, "bold"))
        self.theme_btn.pack(side="right", padx=16)

    def _build_toolbar(self, parent):
        bar = tk.Frame(parent, bg=self.theme["bg_alt"])
        bar.pack(fill="x")
        self._themed_widgets.append((bar, "bg_alt"))

        inner = tk.Frame(bar, bg=self.theme["bg_alt"])
        inner.pack(fill="x", padx=10, pady=8)
        self._themed_widgets.append((inner, "bg_alt"))

        def make_btn(label, command, kind="normal"):
            b = tk.Button(inner, text=label, command=command, relief="flat", cursor="hand2",
                          font=("Segoe UI", 9, "bold"), padx=10, pady=6)
            b.pack(side="left", padx=4)
            self._themed_widgets.append((b, kind))
            return b

        self.btn_upload = make_btn("📂 Upload Image", self.on_upload_image, "accent")
        self.btn_webcam = make_btn("🎥 Open Webcam", self.on_open_webcam, "accent")
        self.btn_stop_webcam = make_btn("⏹ Stop Webcam", self.on_stop_webcam, "danger")
        self.btn_video = make_btn("🎞 Load Video", self.on_load_video, "normal")
        self.btn_detect = make_btn("🔍 Detect", self.on_detect_clicked, "success")
        self.btn_save = make_btn("💾 Save Result", self.on_save_result, "normal")
        self.btn_screenshot = make_btn("📸 Screenshot", self.on_screenshot, "normal")
        self.btn_clear = make_btn("🧹 Clear", self.on_clear, "normal")
        self.btn_exit = make_btn("✕ Exit", self.on_exit, "danger")

        # confidence slider
        conf_box = tk.Frame(inner, bg=self.theme["bg_alt"])
        conf_box.pack(side="left", padx=(20, 4))
        self._themed_widgets.append((conf_box, "bg_alt"))
        self.confidence_label_var = tk.StringVar(value=f"Confidence: {self.confidence:.2f}")
        conf_label = tk.Label(conf_box, textvariable=self.confidence_label_var, font=("Segoe UI", 9),
                               bg=self.theme["bg_alt"], fg=self.theme["fg_muted"])
        conf_label.pack(anchor="w")
        self._themed_widgets.append((conf_label, "bg_alt_fg_muted"))
        self.confidence_slider = ttk.Scale(conf_box, from_=0.10, to=0.90, value=self.confidence,
                                            orient="horizontal", length=150, command=self.on_confidence_change)
        self.confidence_slider.pack()

        self._set_live_controls_state(running=False)

    def _build_preview_area(self, parent):
        left = tk.Frame(parent, bg=self.theme["bg"])
        left.pack(side="left", fill="both", expand=True)
        self._themed_widgets.append((left, "bg"))

        preview_card = tk.Frame(left, bg=self.theme["card"], highlightthickness=1,
                                 highlightbackground=self.theme["border"])
        preview_card.pack(fill="both", expand=True)
        self._themed_widgets.append((preview_card, "card_border"))

        self.preview_canvas = tk.Canvas(preview_card, bg=self.theme["canvas_bg"], highlightthickness=0)
        self.preview_canvas.pack(fill="both", expand=True, padx=2, pady=2)
        self._themed_widgets.append((self.preview_canvas, "canvas"))
        self.preview_canvas.bind("<Configure>", lambda e: self._render_current_frame())
        self.tk_image_ref = None

        zoom_bar = tk.Frame(left, bg=self.theme["bg"])
        zoom_bar.pack(fill="x", pady=(6, 0))
        self._themed_widgets.append((zoom_bar, "bg"))

        def make_zoom_btn(text, cmd):
            b = tk.Button(zoom_bar, text=text, command=cmd, relief="flat", cursor="hand2",
                          font=("Segoe UI", 9, "bold"), padx=8, pady=4)
            b.pack(side="left", padx=3)
            self._themed_widgets.append((b, "normal"))
            return b

        make_zoom_btn("－ Zoom Out", self.on_zoom_out)
        make_zoom_btn("Reset", self.on_zoom_reset)
        make_zoom_btn("＋ Zoom In", self.on_zoom_in)
        make_zoom_btn("⛶ Fullscreen", self.on_fullscreen_preview)

    def _build_sidebar(self, parent):
        sidebar_outer = tk.Frame(parent, bg=self.theme["bg"], width=320)
        sidebar_outer.pack(side="right", fill="y", padx=(12, 0))
        sidebar_outer.pack_propagate(False)
        self._themed_widgets.append((sidebar_outer, "bg"))

        self.scrollable = ScrollableFrame(sidebar_outer, bg=self.theme["bg"])
        self.scrollable.pack(fill="both", expand=True)
        self._themed_widgets.append((self.scrollable, "scrollframe"))
        inner = self.scrollable.inner

        #   statistics card  
        stats_card = tk.Frame(inner, bg=self.theme["card"], highlightthickness=1,
                               highlightbackground=self.theme["border"])
        stats_card.pack(fill="x", pady=(0, 10), ipady=6)
        self._themed_widgets.append((stats_card, "card_border"))

        stats_title = tk.Label(stats_card, text="LIVE STATISTICS", font=("Segoe UI", 10, "bold"),
                                bg=self.theme["card"], fg=self.theme["accent"])
        stats_title.pack(anchor="w", padx=12, pady=(10, 6))
        self._themed_widgets.append((stats_title, "card_accent"))

        stat_defs = [
            ("cars", "Total Cars"),
            ("blue_cars", "Blue Cars"),
            ("other_cars", "Other Cars"),
            ("people", "Total People"),
            ("processing_time", "Processing Time (s)"),
            ("fps", "FPS (webcam/video)"),
        ]
        for key, label in stat_defs:
            row = tk.Frame(stats_card, bg=self.theme["card"])
            row.pack(fill="x", padx=12, pady=3)
            self._themed_widgets.append((row, "card"))
            lbl = tk.Label(row, text=label, font=("Segoe UI", 9), bg=self.theme["card"], fg=self.theme["fg_muted"])
            lbl.pack(side="left")
            self._themed_widgets.append((lbl, "card_fg_muted"))
            var = tk.StringVar(value="0")
            self.stats_vars[key] = var
            val = tk.Label(row, textvariable=var, font=("Consolas", 11, "bold"), bg=self.theme["card"],
                           fg=self.theme["fg"])
            val.pack(side="right")
            self._themed_widgets.append((val, "card_fg"))

        #   recent files quick list  
        recent_card = tk.Frame(inner, bg=self.theme["card"], highlightthickness=1,
                                highlightbackground=self.theme["border"])
        recent_card.pack(fill="x", pady=(0, 10))
        self._themed_widgets.append((recent_card, "card_border"))
        recent_title = tk.Label(recent_card, text="RECENT FILES", font=("Segoe UI", 10, "bold"),
                                 bg=self.theme["card"], fg=self.theme["accent"])
        recent_title.pack(anchor="w", padx=12, pady=(10, 6))
        self._themed_widgets.append((recent_title, "card_accent"))
        self.recent_listbox = tk.Listbox(recent_card, height=4, font=("Segoe UI", 9), relief="flat",
                                          bg=self.theme["entry_bg"], fg=self.theme["fg"],
                                          highlightthickness=0, selectbackground=self.theme["accent"])
        self.recent_listbox.pack(fill="x", padx=12, pady=(0, 10))
        self.recent_listbox.bind("<Double-Button-1>", self._on_recent_listbox_click)
        self._themed_widgets.append((self.recent_listbox, "listbox"))

        #   detection history  
        history_card = tk.Frame(inner, bg=self.theme["card"], highlightthickness=1,
                                 highlightbackground=self.theme["border"])
        history_card.pack(fill="x", pady=(0, 10))
        self._themed_widgets.append((history_card, "card_border"))
        history_title = tk.Label(history_card, text="DETECTION HISTORY", font=("Segoe UI", 10, "bold"),
                                  bg=self.theme["card"], fg=self.theme["accent"])
        history_title.pack(anchor="w", padx=12, pady=(10, 6))
        self._themed_widgets.append((history_title, "card_accent"))
        self.history_listbox = tk.Listbox(history_card, height=8, font=("Consolas", 8), relief="flat",
                                           bg=self.theme["entry_bg"], fg=self.theme["fg"],
                                           highlightthickness=0, selectbackground=self.theme["accent"])
        self.history_listbox.pack(fill="x", padx=12, pady=(0, 10))
        self._themed_widgets.append((self.history_listbox, "listbox"))

    def _build_statusbar(self, parent):
        bar = tk.Frame(parent, bg=self.theme["toolbar"], height=28)
        bar.pack(fill="x", side="bottom")
        self._themed_widgets.append((bar, "toolbar"))
        lbl = tk.Label(bar, textvariable=self.status_var, font=("Segoe UI", 9), bg=self.theme["toolbar"],
                       fg=self.theme["fg_muted"], anchor="w")
        lbl.pack(side="left", padx=10, pady=4)
        self._themed_widgets.append((lbl, "toolbar_fg_muted"))

    def _show_about(self):
        messagebox.showinfo(
            "About",
            f"{self.APP_TITLE}\n\n"
            "YOLOv8 + OpenCV + Tkinter desktop application.\n"
            "Detects cars & people, classifies car colour via HSV + K-Means,\n"
            "and reports live traffic statistics.",
        )

                  
    def on_toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.theme = utils.DARK_THEME if self.dark_mode else utils.LIGHT_THEME
        self._apply_theme()

    def _apply_theme(self):
        t = self.theme
        self.root.configure(bg=t["bg"])
        self.theme_btn.config(text="☀ Light Mode" if self.dark_mode else "🌙 Dark Mode",
                               bg=t["accent"], fg=t["accent_fg"], activebackground=t["accent"])

        role_bg = {
            "bg": t["bg"], "bg_alt": t["bg_alt"], "toolbar": t["toolbar"],
            "card": t["card"], "canvas": t["canvas_bg"], "listbox": t["entry_bg"],
        }
        for widget, kind in self._themed_widgets:
            try:
                if kind in role_bg:
                    widget.configure(bg=role_bg[kind])
                    if kind == "listbox":
                        widget.configure(fg=t["fg"], selectbackground=t["accent"])
                elif kind == "card_border":
                    widget.configure(bg=t["card"], highlightbackground=t["border"])
                elif kind == "toolbar_fg":
                    widget.configure(bg=t["toolbar"], fg=t["fg"])
                elif kind == "toolbar_fg_muted":
                    widget.configure(bg=t["toolbar"], fg=t["fg_muted"])
                elif kind == "bg_alt_fg_muted":
                    widget.configure(bg=t["bg_alt"], fg=t["fg_muted"])
                elif kind == "card_fg":
                    widget.configure(bg=t["card"], fg=t["fg"])
                elif kind == "card_fg_muted":
                    widget.configure(bg=t["card"], fg=t["fg_muted"])
                elif kind == "card_accent":
                    widget.configure(bg=t["card"], fg=t["accent"])
                elif kind == "scrollframe":
                    widget.set_bg(t["bg"])
                elif kind == "accent":
                    widget.configure(bg=t["accent"], fg=t["accent_fg"], activebackground=t["info"])
                elif kind == "success":
                    widget.configure(bg=t["success"], fg="#ffffff", activebackground=t["success"])
                elif kind == "danger":
                    widget.configure(bg=t["danger"], fg="#ffffff", activebackground=t["danger"])
                elif kind == "normal":
                    widget.configure(bg=t["border"], fg=t["fg"], activebackground=t["info"])
            except tk.TclError:
                pass  # widget already destroyed - safe to skip

        self._render_current_frame()

    #                  
    # Frame processing (shared by static Detect + live loops)
    #                  
    def _process_frame(self, frame_bgr):
        return detector.annotate_frame(frame_bgr, self.detector, self.color_detector)

    def _update_stats_labels(self, stats, fps=None):
        self.stats_vars["cars"].set(str(stats.get("cars", 0)))
        self.stats_vars["blue_cars"].set(str(stats.get("blue_cars", 0)))
        self.stats_vars["other_cars"].set(str(stats.get("other_cars", 0)))
        self.stats_vars["people"].set(str(stats.get("people", 0)))
        self.stats_vars["processing_time"].set(f"{stats.get('processing_time', 0):.3f}")
        if fps is not None:
            self.stats_vars["fps"].set(f"{fps:.1f}")

    def _reset_stats_labels(self):
        for key in ("cars", "blue_cars", "other_cars", "people"):
            self.stats_vars[key].set("0")
        self.stats_vars["processing_time"].set("0.000")
        self.stats_vars["fps"].set("0")

                    
    def on_upload_image(self):
        if self.webcam_running or self.video_running:
            messagebox.showinfo("Live Mode Active", "Stop the webcam/video first, then upload an image.")
            return
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
            initialdir=utils.IMAGES_DIR,
        )
        if not path:
            return
        image = utils.imread_safe(path)
        if image is None:
            messagebox.showerror("Invalid Image", "The selected file could not be read as a valid image.\n"
                                                    "It may be corrupted or in an unsupported format.")
            return
        self.image_path = path
        self.current_image_bgr = image
        self.display_image_bgr = image
        self.zoom_factor = 1.0
        self._reset_stats_labels()
        self._render_current_frame()
        self.status_var.set(f"Loaded: {os.path.basename(path)}  ({image.shape[1]}×{image.shape[0]})")
        utils.add_recent_file(path)
        self._refresh_recent_files_menu()

    def on_detect_clicked(self):
        if self.webcam_running or self.video_running:
            messagebox.showinfo("Live Mode Active", "Detection is already running live on the webcam/video feed.")
            return
        if self.current_image_bgr is None:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return
        if self.detector is None or not self.detector.is_ready:
            messagebox.showerror("Model Unavailable", "The YOLOv8 model could not be loaded, so detection is disabled.")
            return

        self.status_var.set("Processing…")
        self.root.update_idletasks()

        annotated, stats, details = self._process_frame(self.current_image_bgr)
        self.display_image_bgr = annotated
        self._update_stats_labels(stats)
        self._render_current_frame()

        if stats["cars"] == 0 and stats["people"] == 0:
            self.status_var.set("Detection Completed — no cars or people found in this image")
        else:
            self.status_var.set("Detection Completed")

        utils.add_history_entry({
            "image_name": os.path.basename(self.image_path) if self.image_path else "unsaved_frame",
            "cars": stats["cars"], "blue_cars": stats["blue_cars"],
            "other_cars": stats["other_cars"], "people": stats["people"],
            "processing_time": stats["processing_time"],
        })
        self._refresh_history_list()

    def on_save_result(self):
        if self.display_image_bgr is None:
            messagebox.showwarning("Nothing to Save", "There is no image to save yet. Upload an image (and run "
                                                        "Detect) first.")
            return
        default_name = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = filedialog.asksaveasfilename(
            title="Save annotated image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
            initialdir=utils.OUTPUTS_DIR,
            initialfile=default_name,
        )
        if not path:
            return
        if not utils.imwrite_safe(path, self.display_image_bgr):
            messagebox.showerror("Save Failed", "Could not save the annotated image to that location.")
            return

        utils.append_csv_report(utils.CSV_REPORT_PATH, {
            "Image Name": os.path.basename(self.image_path) if self.image_path else os.path.basename(path),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Cars": self.stats_vars["cars"].get(),
            "Blue Cars": self.stats_vars["blue_cars"].get(),
            "Other Cars": self.stats_vars["other_cars"].get(),
            "People": self.stats_vars["people"].get(),
            "Processing Time (s)": self.stats_vars["processing_time"].get(),
        })
        self.status_var.set(f"Saved: {os.path.basename(path)}  (statistics appended to detection_report.csv)")

    def on_screenshot(self):
        img = self.display_image_bgr if self.display_image_bgr is not None else self.current_image_bgr
        if img is None:
            messagebox.showwarning("Nothing to Capture", "Start the webcam or load an image first.")
            return
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(utils.OUTPUTS_DIR, filename)
        if utils.imwrite_safe(path, img):
            self.status_var.set(f"Screenshot saved: {filename}")
        else:
            messagebox.showerror("Screenshot Failed", "Could not save the screenshot.")

    def on_clear(self):
        self.current_image_bgr = None
        self.display_image_bgr = None
        self.image_path = None
        self.zoom_factor = 1.0
        self._reset_stats_labels()
        self._render_current_frame()
        self.status_var.set("Ready")

    def on_exit(self):
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            self.webcam_running = False
            self.video_running = False
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.root.destroy()

                   
    def on_open_webcam(self):
        if self.webcam_running or self.video_running:
            return
        if self.detector is None or not self.detector.is_ready:
            messagebox.showerror("Model Unavailable", "The YOLOv8 model could not be loaded, so live detection is disabled.")
            return
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "No camera detected. Please check your webcam connection and try again.")
            cap.release()
            return
        self.cap = cap
        self.webcam_running = True
        self._fps_history.clear()
        self._set_live_controls_state(running=True)
        self.status_var.set("Live detection running…")
        self._capture_thread = threading.Thread(target=self._capture_loop, args=("webcam",), daemon=True)
        self._capture_thread.start()
        self.root.after(15, self._poll_frame_queue)

    def on_stop_webcam(self):
        if not (self.webcam_running or self.video_running):
            return
        self.webcam_running = False
        self.video_running = False
        self._set_live_controls_state(running=False)
        self.status_var.set("Webcam stopped")

    def on_load_video(self):
        if self.webcam_running or self.video_running:
            messagebox.showinfo("Live Mode Active", "Stop the current webcam/video before loading another.")
            return
        if self.detector is None or not self.detector.is_ready:
            messagebox.showerror("Model Unavailable", "The YOLOv8 model could not be loaded, so video detection is disabled.")
            return
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Invalid Video", "The selected file could not be opened as a video.")
            cap.release()
            return
        self.cap = cap
        self.video_running = True
        self._fps_history.clear()
        self._set_live_controls_state(running=True)
        self.status_var.set(f"Playing: {os.path.basename(path)}")
        self._capture_thread = threading.Thread(target=self._capture_loop, args=("video",), daemon=True)
        self._capture_thread.start()
        self.root.after(15, self._poll_frame_queue)

    def _capture_loop(self, source_kind):
        """Runs on a background thread. Never touches Tkinter widgets -
        only pushes results onto self.frame_queue."""
        prev_time = time.time()
        source_fps = None
        if source_kind == "video" and self.cap is not None:
            reported = self.cap.get(cv2.CAP_PROP_FPS)
            source_fps = reported if reported and reported > 1 else 25.0
        frame_delay = (1.0 / source_fps) if source_fps else 0.0

        running_flag = "webcam_running" if source_kind == "webcam" else "video_running"
        while getattr(self, running_flag):
            ok, frame = self.cap.read()
            if not ok:
                break
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            annotated, stats, _details = self._process_frame(frame)
            try:
                self.frame_queue.put_nowait((annotated, stats, fps))
            except queue.Full:
                pass  # UI hasn't kept up - drop this frame, not fatal

            if frame_delay:
                time.sleep(max(0.0, frame_delay - (time.time() - now)))

        if self.cap is not None:
            self.cap.release()
            self.cap = None
        setattr(self, running_flag, False)

    def _poll_frame_queue(self):
        if not (self.webcam_running or self.video_running):
            
            return
        latest = None
        try:
            while True:
                latest = self.frame_queue.get_nowait()
        except queue.Empty:
            pass

        if latest is not None:
            annotated, stats, fps = latest
            self.display_image_bgr = annotated
            self._fps_history.append(fps)
            avg_fps = sum(self._fps_history) / len(self._fps_history)
            self._update_stats_labels(stats, fps=avg_fps)
            self._render_current_frame()

        self.root.after(15, self._poll_frame_queue)

    def _set_live_controls_state(self, running):
        live_state = "disabled" if running else "normal"
        static_state = "normal" if running else "disabled"
        for b in (self.btn_upload, self.btn_video, self.btn_detect, self.btn_webcam):
            b.config(state=live_state)
        self.btn_stop_webcam.config(state=static_state)

                   
    def on_zoom_in(self):
        self.zoom_factor = min(self.zoom_factor * 1.2, 4.0)
        self._render_current_frame()

    def on_zoom_out(self):
        self.zoom_factor = max(self.zoom_factor / 1.2, 0.2)
        self._render_current_frame()

    def on_zoom_reset(self):
        self.zoom_factor = 1.0
        self._render_current_frame()

    def on_fullscreen_preview(self):
        img = self.display_image_bgr if self.display_image_bgr is not None else self.current_image_bgr
        if img is None:
            messagebox.showwarning("No Image", "There is no image to preview yet.")
            return
        top = tk.Toplevel(self.root)
        top.title("Fullscreen Preview  (Esc to close)")
        top.configure(bg="black")
        top.attributes("-fullscreen", True)
        top.bind("<Escape>", lambda e: top.destroy())

        top.update_idletasks()
        screen_w, screen_h = top.winfo_screenwidth(), top.winfo_screenheight()
        resized = utils.resize_to_fit(img, screen_w, screen_h)
        tk_img = utils.bgr_to_tk_image(resized)
        label = tk.Label(top, image=tk_img, bg="black")
        label.image = tk_img  # keep a reference alive
        label.pack(expand=True)

        close_btn = tk.Button(top, text="✕ Close", command=top.destroy, bg="#ef4444", fg="white",
                              relief="flat", font=("Segoe UI", 10, "bold"), padx=10, pady=4)
        close_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-16, y=16)

    #                  
    # Rendering
    #                  
    def _render_current_frame(self):
        if not hasattr(self, "preview_canvas"):
            return
        img = self.display_image_bgr if self.display_image_bgr is not None else self.current_image_bgr
        self.preview_canvas.delete("all")

        canvas_w = max(self.preview_canvas.winfo_width(), 50)
        canvas_h = max(self.preview_canvas.winfo_height(), 50)

        if img is None:
            self.preview_canvas.create_text(
                canvas_w // 2, canvas_h // 2, text="No image loaded\nUpload an image or start the webcam",
                fill=self.theme["fg_muted"], font=("Segoe UI", 12), justify="center",
            )
            return

        base = utils.resize_to_fit(img, max(canvas_w - 24, 10), max(canvas_h - 24, 10), allow_upscale=False)
        if abs(self.zoom_factor - 1.0) > 1e-3:
            h, w = base.shape[:2]
            new_w, new_h = max(1, int(w * self.zoom_factor)), max(1, int(h * self.zoom_factor))
            base = cv2.resize(base, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        tk_img = utils.bgr_to_tk_image(base)
        self.tk_image_ref = tk_img  # prevent garbage collection
        self.preview_canvas.create_image(canvas_w // 2, canvas_h // 2, image=tk_img, anchor="center")

                     
    def on_confidence_change(self, value):
        conf = float(value)
        self.confidence = conf
        if self.detector:
            self.detector.set_confidence(conf)
        self.confidence_label_var.set(f"Confidence: {conf:.2f}")

                    
    def _refresh_recent_files_menu(self):
        self.recent_menu.delete(0, "end")
        self.recent_listbox.delete(0, "end")
        files = utils.load_recent_files()
        if not files:
            self.recent_menu.add_command(label="(No recent files)", state="disabled")
            self.recent_listbox.insert("end", "  (No recent files yet)")
            return
        for path in files:
            self.recent_menu.add_command(label=os.path.basename(path), command=lambda p=path: self._load_recent(p))
            self.recent_listbox.insert("end", "  " + os.path.basename(path))
        self._recent_paths = files

    def _on_recent_listbox_click(self, _event):
        sel = self.recent_listbox.curselection()
        if not sel or not getattr(self, "_recent_paths", None):
            return
        idx = sel[0]
        if idx < len(self._recent_paths):
            self._load_recent(self._recent_paths[idx])

    def _load_recent(self, path):
        if self.webcam_running or self.video_running:
            messagebox.showinfo("Live Mode Active", "Stop the webcam/video first.")
            return
        if not os.path.isfile(path):
            messagebox.showwarning("File Not Found", f"{path}\n\nThis file no longer exists.")
            return
        image = utils.imread_safe(path)
        if image is None:
            messagebox.showerror("Invalid Image", "This file could not be read as a valid image.")
            return
        self.image_path = path
        self.current_image_bgr = image
        self.display_image_bgr = image
        self.zoom_factor = 1.0
        self._reset_stats_labels()
        self._render_current_frame()
        self.status_var.set(f"Loaded: {os.path.basename(path)}")
        utils.add_recent_file(path)
        self._refresh_recent_files_menu()

    def _refresh_history_list(self):
        self.history_listbox.delete(0, "end")
        history = utils.load_history()
        if not history:
            self.history_listbox.insert("end", "  (No detections yet)")
            return
        for entry in history:
            line = (f"  {entry.get('timestamp', '')}  "
                     f"C:{entry.get('cars', 0)} B:{entry.get('blue_cars', 0)} "
                     f"P:{entry.get('people', 0)}")
            self.history_listbox.insert("end", line)
