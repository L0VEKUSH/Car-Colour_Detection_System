
import sys
import tkinter as tk
from tkinter import messagebox

import utils


def main():
    utils.ensure_app_dirs()

    root = tk.Tk()
    root.title("Car Colour Detection and Traffic Monitoring System")
    try:
        root.state("zoomed")  # maximised window on Windows
    except tk.TclError:
        root.geometry("1400x860")
    root.minsize(1100, 650)

 
    from gui import CarColorDetectionApp

    try:
        app = CarColorDetectionApp(root)
    except Exception as exc:  # top-level startup guard
        messagebox.showerror("Startup Error", f"The application failed to start:\n\n{exc}")
        sys.exit(1)

    root.protocol("WM_DELETE_WINDOW", app.on_exit)
    root.mainloop()


if __name__ == "__main__":
    main()
