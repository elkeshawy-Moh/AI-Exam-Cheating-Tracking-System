# AI Exam Cheating Tracking and Alert System 🚨🎓

An advanced Desktop AI system built using Python to track students during exams, automatically detect suspicious cheating behaviors, and trigger real-time local desktop alerts and permanent tracking logs.

## ✨ Key Features
- **Intelligent Object Tracking:** Powered by **YOLOv8** and **ByteTrack** for high-precision multiple-student detection.
- **Cheating Detection Rules:** Captures advanced behaviors including Peeking/Looking around, Proximity Violations (whispering/paper swapping), and illegal Cell Phone Usage.
- **Dynamic Bounding Boxes:** Instant UI updates that switch a suspect's box from **Green** to **Red**.
- **Permanent Alert History:** Logs every cheating event automatically with Date/Time, Behavior type, and AI Confidence Scores into a structured local database.
- **Optional Screenshot Viewer:** Built-in Tkinter function allowing invigilators to select a log and instantly open the annotated screenshot with a single click.

## 🛠️ Tech Stack & Architecture
- **AI Core:** Ultralytics YOLOv8 (yolov8n.pt)
- **GUI Engine:** Python Tkinter Interface
- **Image Processing:** OpenCV & Pillow
- **Database System:** Structural Local JSON Logging Architecture

## 🚀 How to Run Locally

1. **Clone the Repository:**
   ```bash
   git clone https://github.com
   cd AI-Exam-Cheating-Tracking-System
   ```

2. **Install Required Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add Missing Model Files:**
   Download `yolov8n.pt` and place it directly into the root folder.

4. **Launch the Application:**
   ```bash
   python main_app.py
   ```
