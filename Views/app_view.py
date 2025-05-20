from Controllers.app_controller import start_sfm_process
import tkinter as tk
from tkinter import ttk

def create_main_window():
    #adel all
    # Main window
    root = tk.Tk()
    root.title("Stereo Processing UI")
    root.geometry("600x400")

    # Bundle Adjustment dropdown
    tk.Label(root, text="Bundle Adjustment").pack(pady=5)
    ba_var = tk.StringVar(value="Yes")
    ba_dropdown = ttk.Combobox(root, textvariable=ba_var, values=["Yes", "No"], state="readonly")
    ba_dropdown.pack()

    # Data type dropdown
    tk.Label(root, text="Dataset").pack(pady=5)
    data_var = tk.StringVar(value="Calibrate")
    data_dropdown = ttk.Combobox(root, textvariable=data_var, values=["Calibrate", "Pre-calibrated"], state="readonly")
    data_dropdown.pack()

    tk.Label(root, text="Show Corners Detection").pack(pady=5)
    corner_var = tk.StringVar(value="No")
    corner_dropdown = ttk.Combobox(root, textvariable=corner_var, values=["No", "Yes"], state="readonly")
    corner_dropdown.pack()

    tk.Label(root, text="Show matches").pack(pady=5)
    matches_var = tk.StringVar(value="No")
    matches_dropdown = ttk.Combobox(root, textvariable=matches_var, values=["No", "Yes"], state="readonly")
    matches_dropdown.pack()
    def clicked():
        start_sfm_process(ba_var,data_var,matches_var,corner_var)
    # Start button
    tk.Button(root, text="Start", command=clicked).pack(pady=20)

    root.mainloop()
