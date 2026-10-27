
import os
import cv2
import json
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO

# Get the current directory path automatically
current_dir = os.path.dirname(os.path.abspath(__file__))

# --- LOCAL DATABASE SETUP (JSON FILE) ---
# We will use a local JSON file to save all cheating history permanently
history_file = os.path.join(current_dir, "alerts", "cheating_history.json")
screenshot_dir = os.path.join(current_dir, "alerts", "screenshots")

os.makedirs(screenshot_dir, exist_ok=True)

# Load the local model weight file
model_weight_path = os.path.join(current_dir, "yolov8n.pt")
model = YOLO(model_weight_path)

# Track captured cheating instances in the current session
captured_cheaters = set()


class CheatingDetectionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("AI Exam Cheating Tracking and Alert System")
        self.window.geometry("1200x750")
        self.window.configure(bg="#2c3e50")

        self.cap = None

        # 1. UI Header Layout
        self.header_label = tk.Label(
            window, 
            text="AI Exam Cheating Tracking and Alert System", 
            font=("Helvetica", 20, "bold"), 
            bg="#34495e", 
            fg="white", 
            pady=12
        )
        self.header_label.pack(fill=tk.X, side=tk.TOP)

        # Bottom Control Panel Row
        self.btn_frame = tk.Frame(window, bg="#2c3e50")
        self.btn_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=20)

        self.upload_btn = tk.Button(
            self.btn_frame, 
            text="📁 Upload Exam Video", 
            font=("Helvetica", 12, "bold"), 
            bg="#2ecc71", 
            fg="white", 
            padx=20, 
            pady=8, 
            command=self.upload_video
        )
        self.upload_btn.pack(side=tk.LEFT, padx=40)

        self.view_btn = tk.Button(
            self.btn_frame, 
            text="🖼️ Open Screenshots Folder", 
            font=("Helvetica", 12, "bold"), 
            bg="#3498db", 
            fg="white", 
            padx=20, 
            pady=8, 
            command=self.open_screenshot_folder
        )
        self.view_btn.pack(side=tk.RIGHT, padx=40)

        # Optional [Screenshot Viewer] Button for selected rows
        self.open_selected_btn = tk.Button(
            self.btn_frame, 
            text="🔍 View Selected Incident Photo", 
            font=("Helvetica", 12, "bold"), 
            bg="#e67e22", 
            fg="white", 
            padx=20, 
            pady=8, 
            command=self.view_selected_screenshot
        )
        self.open_selected_btn.pack(side=tk.BOTTOM, pady=5)

        # Main Workspace Container split into Left (Video) and Right (History logs)
        self.main_frame = tk.Frame(window, bg="#2c3e50")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Left Column: Video Workspace Panel
        self.left_frame = tk.Frame(self.main_frame, bg="#2c3e50")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_display = tk.Label(self.left_frame, bg="black", width=65, height=22)
        self.video_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Right Column: Alert History Log panel
        self.right_frame = tk.Frame(self.main_frame, bg="#34495e", width=420)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)

        self.history_title = tk.Label(
            self.right_frame, 
            text="📜 Permanent Alert History", 
            font=("Helvetica", 14, "bold"), 
            bg="#34495e", 
            fg="#e74c3c"
        )
        self.history_title.pack(pady=10)

        # Treeview list layout columns (Date/Time, ID, Behavior, Conf)
        self.history_tree = ttk.Treeview(self.right_frame, columns=("Time", "ID", "Behavior", "Conf"), show="headings", height=15)
        self.history_tree.heading("Time", text="Date/Time")
        self.history_tree.heading("ID", text="Student ID")
        self.history_tree.heading("Behavior", text="Behavior")
        self.history_tree.heading("Conf", text="Conf %")
        
        self.history_tree.column("Time", width=130, anchor=tk.CENTER)
        self.history_tree.column("ID", width=70, anchor=tk.CENTER)
        self.history_tree.column("Behavior", width=120, anchor=tk.W)
        self.history_tree.column("Conf", width=60, anchor=tk.CENTER)
        self.history_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Dictionary to hold the path of each item in the treeview for the viewer button
        self.tree_paths = {}

        # Load saved history from JSON file directly into UI when the application opens
        self.load_history_from_file()

    def upload_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv")]
        )
        if file_path:
            video_file_name = os.path.basename(file_path)
            print(f"🎬 Processing file selection stream: {video_file_name}")
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(file_path)
            self.process_video(video_file_name)

    def open_screenshot_folder(self):
        os.startfile(screenshot_dir)
    def save_incident_to_local_db(self, time_str, track_id, reason, conf, screenshot_path):
        """Requirement 4: Save every cheating event locally to JSON database file"""
        new_incident = {
            "datetime": time_str,
            "student_id": int(track_id),
            "behavior": str(reason),
            "confidence_score": f"{float(conf)*100:.1f}%",
            "screenshot_path": str(screenshot_path)
        }
        
        # Read current data from file if it exists
        data = []
        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as file:
                    data = json.load(file)
            except Exception:
                data = []
                
        # Append and write back to file
        data.append(new_incident)
        with open(history_file, "w") as file:
            json.dump(data, file, indent=4)
        print(f"💾 LOCAL DB: Incident logged completely in JSON history file.")

    def load_history_from_file(self):
        """Loads saved logs back into Treeview when starting up"""
        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as file:
                    data = json.load(file)
                    for item in data:
                        item_id = self.history_tree.insert(
                            "", 0, 
                            values=(item["datetime"], f"ID: {item['student_id']}", item["behavior"], item["confidence_score"])
                        )
                        # Link item ID to its screenshot path
                        self.tree_paths[item_id] = item["screenshot_path"]
            except Exception as e:
                print(f"Failed to read local history file: {e}")

    def view_selected_screenshot(self):
        """Optional [Screenshot Viewer]: Opens the saved image for the selected table row"""
        selected_item = self.history_tree.selection()
        if not selected_item:
            messagebox.showwarning("Selection Required", "Please click on a row in the Alert History table first!")
            return
            
        item_id = selected_item[0]
        img_path = self.tree_paths.get(item_id, "")
        
        if img_path and os.path.exists(img_path):
            os.startfile(img_path)
        else:
            messagebox.showerror("Error", "Screenshot file not found or path missing.")

    def process_video(self, video_file_name):
        if self.cap and self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                video_height, video_width = frame.shape[:2]
                center_line = video_width // 2

                # Run AI tracking engine
                results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0, 67], conf=0.15, verbose=False)
                
                # FIX: Safeguard extraction from results list object to prevent AttributeError
                if len(results) > 0 and results[0] is not None:
                    result_item = results[0]
                else:
                    result_item = None

                display_frame = frame.copy()
                students = {}
                phones = []

                if result_item is not None and result_item.boxes is not None:
                    boxes = result_item.boxes.xyxy.cpu().numpy()
                    clss = result_item.boxes.cls.cpu().numpy().astype(int)
                    confs = result_item.boxes.conf.cpu().numpy()
                    track_ids = result_item.boxes.id.cpu().numpy().astype(int) if result_item.boxes.id is not None else [None] * len(boxes)

                    for box, cls, conf, track_id in zip(boxes, clss, confs, track_ids):
                        x1, y1, x2, y2 = map(int, box)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        box_height = y2 - y1
                        if cls == 0 and y1 < video_height * 0.15 and box_height > video_height * 0.6:
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(display_frame, "INVIGILATOR", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            continue

                        if cls == 0 and track_id is not None:
                            students[track_id] = {
                                'box': (x1, y1, x2, y2), 
                                'center': center_x, 
                                'is_cheating': False, 
                                'reason': "", 
                                'conf': conf
                            }
                        elif cls == 67 and (x2 - x1) > 20:
                            phones.append((x1, y1, x2, y2, center_x, center_y, conf))

                    # Rule 1: Adaptive Peeking Detection
                    for track_id, data in students.items():
                        x1, y1, x2, y2 = data['box']
                        if data['center'] < center_line and ((x2 - x1) / (y2 - y1)) > 0.85:
                            students[track_id]['is_cheating'] = True
                            students[track_id]['reason'] = "Peeking Behavior"

                    # Rule 2: Proximity Verification
                    student_ids = list(students.keys())
                    for i in range(len(student_ids)):
                        for j in range(i + 1, len(student_ids)):
                            id1, id2 = student_ids[i], student_ids[j]
                            spatial_dist = abs(students[id1]['center'] - students[id2]['center'])
                            if spatial_dist < 145:
                                students[id1]['is_cheating'] = True
                                students[id2]['is_cheating'] = True
                                students[id1]['reason'] = "Proximity Violation"
                                students[id2]['reason'] = "Proximity Violation"

                    # Rule 3: Mobile phone interaction check
                    for p_box in phones:
                        px1, py1, px2, py2, pcx, pcy, p_conf = p_box
                        cv2.rectangle(display_frame, (px1, py1), (px2, py2), (0, 0, 255), 3)
                        
                        closest_student_id = None
                        min_dist = float('inf')
                        for track_id, data in students.items():
                            dist_to_phone = abs(data['center'] - pcx)
                            if dist_to_phone < min_dist:
                                min_dist = dist_to_phone
                                closest_student_id = track_id
                        
                        if closest_student_id is not None and min_dist < 260:
                            students[closest_student_id]['is_cheating'] = True
                            students[closest_student_id]['reason'] = "Phone Usage"

                    # Draw layout and trigger alerts
                    for track_id, data in students.items():
                        x1, y1, x2, y2 = data['box']
                        if data['is_cheating']:
                            box_color = (0, 0, 255)
                            label = f"Student ID: {track_id} [{data['reason']}]"
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 4)
                            cv2.putText(display_frame, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                            alert_key = f"{video_file_name}_{track_id}_{data['reason']}"
                            if alert_key not in captured_cheaters:
                                current_time = datetime.now()
                                time_str = current_time.strftime("%Y-%m-%d %I:%M %p")
                                stamp_file = current_time.strftime("%Y%m%d_%H%M%S")
                                
                                screenshot_name = f"cheat_{video_file_name[:-4]}_id_{track_id}_{stamp_file}.jpg"
                                screenshot_path = os.path.join(screenshot_dir, screenshot_name)
                                cv2.imwrite(screenshot_path, display_frame)
                                
                                # Instantly append event to the table
                                conf_pct = f"{float(data['conf'])*100:.1f}%"
                                item_id = self.history_tree.insert("", 0, values=(time_str, f"ID: {track_id}", data['reason'], conf_pct))
                                self.tree_paths[item_id] = screenshot_path

                                # Save to permanent local JSON database file
                                self.save_incident_to_local_db(time_str, track_id, data['reason'], data['conf'], screenshot_path)
                                    
                                captured_cheaters.add(alert_key)
                                
                                # Notification System Trigger Popup
                                messagebox.showwarning(
                                    "🚨 Cheating Alert Detected",
                                    f"Behavior: {data['reason']}\nTime: {current_time.strftime('%I:%M %p')}\n\nScreenshot saved successfully."
                                )
                        else:
                            box_color = (0, 255, 0)
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 4)
                            cv2.putText(display_frame, f"Student ID: {track_id}", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(display_frame)
                img = img.resize((700, 450), Image.Resampling.LANCZOS)

                img_tk = ImageTk.PhotoImage(image=img)

                self.video_display.img_tk = img_tk
                self.video_display.configure(image=img_tk)
                self.window.after(10, lambda: self.process_video(video_file_name))
            else:
                self.cap.release()
                messagebox.showinfo("Processing Done", f"Successfully completed checking: {video_file_name}")

# Launch desktop interface safely
if __name__ == "__main__":
    root = tk.Tk()
    app = CheatingDetectionApp(root)
    root.mainloop()
