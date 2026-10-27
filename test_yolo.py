import os
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# --- FIREBASE INITIALIZATION ---
import firebase_admin
from firebase_admin import credentials, firestore

# Get the current directory path automatically
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define path to your downloaded Firebase private key
key_path = os.path.join(current_dir, "firebase_key.json")

# Initialize Firebase Admin SDK safely
if not firebase_admin._apps:
    cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred)

# Initialize Cloud Firestore Database client
db = firestore.client()
print("🔥 Firebase Connected Successfully!")

# --- LOCAL DIRECTORY SETUP ---
# Create the locally required screenshot folder structure
screenshot_dir = os.path.join(current_dir, "alerts", "screenshots")
os.makedirs(screenshot_dir, exist_ok=True)

# Find ALL .mp4 video files inside the current folder
video_files = [f for f in os.listdir(current_dir) if f.lower().endswith('.mp4')]

if not video_files:
    print("No video files found in this folder!")
    exit()

print(f"📚 Found {len(video_files)} video files to process: {video_files}")

# Load the local model weight file
model_weight_path = os.path.join(current_dir, "yolov8n.pt")
model = YOLO(model_weight_path)

# A set to keep track of captured cheating instances globally
captured_cheaters = set()

# Loop through each video file found in the folder one by one
for video_file in video_files:
    video_path = os.path.join(current_dir, video_file)
    print(f"\n🚀 Launching system on: {video_file}")
    
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print(f"Finished processing video: {video_file}")
            break

        # Get video frame dimensions
        video_height, video_width = frame.shape[:2]
        center_line = video_width // 2

        # Track both Persons (0) and Cell Phones (67)
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0, 67], conf=0.15, verbose=False)

        display_frame = frame.copy()

        # Dictionaries to keep track of entities on the current frame
        students = {}
        phones = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else [None] * len(boxes)

            for box, cls, track_id in zip(boxes, clss, track_ids):
                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Exclude the supervisor/invigilator based on their dynamic vertical position/height (standing)
                box_height = y2 - y1
                if cls == 0 and y1 < video_height * 0.15 and box_height > video_height * 0.6:
                    # Draw a unique BLUE bounding box for the invigilator/supervisor to keep them out of cheating logic
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(display_frame, "INVIGILATOR", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    continue

                if cls == 0 and track_id is not None:  # Confirmed Student
                    students[track_id] = {
                        'box': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'is_cheating': False,
                        'reason': ""
                    }
                elif cls == 67:  # Cell Phone
                    # Size filter: Exclude small boxes to prevent mistaking watches/clocks as phones
                    phone_width = x2 - x1
                    if phone_width > 20:  # Adjust thickness/pixel threshold based on dataset
                        phones.append((x1, y1, x2, y2, center_x, center_y))

            # --- CHEATING DETECTION RULESETS ---
            # RULE 1: Adaptive Peeking Detection (Checks bounding box stretch and horizontal shifting)
            for track_id, data in students.items():
                x1, y1, x2, y2 = data['box']
                aspect_ratio = (x2 - x1) / (y2 - y1)
                # If a student leans dramatically or turns head sharply, their box distorts on the left side
                if data['center'][0] < center_line and aspect_ratio > 0.85:
                    students[track_id]['is_cheating'] = True
                    students[track_id]['reason'] = "Peeking Behavior"

            # RULE 2: Proximity Check (Students whispering or exchanging papers)
            student_ids = list(students.keys())
            for i in range(len(student_ids)):
                for j in range(i + 1, len(student_ids)):
                    id1, id2 = student_ids[i], student_ids[j]
                    c1 = np.array(students[id1]['center'])
                    c2 = np.array(students[id2]['center'])
                    spatial_dist = np.linalg.norm(c1 - c2)
                    
                    if spatial_dist < 145:  # Trigger proximity violation alert
                        students[id1]['is_cheating'] = True
                        students[id2]['is_cheating'] = True
                        students[id1]['reason'] = "Proximity Violation"
                        students[id2]['reason'] = "Proximity Violation"

            # RULE 3: Proximity Phone Usage Check
            for p_box in phones:
                px1, py1, px2, py2, pcx, pcy = p_box
                cv2.rectangle(display_frame, (px1, py1), (px2, py2), (0, 0, 255), 3)
                cv2.putText(display_frame, "PHONE DETECTED", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                closest_student_id = None
                min_dist = float('inf')
                for track_id, data in students.items():
                    scx, scy = data['center']
                    dist_to_phone = np.linalg.norm(np.array([scx, scy]) - np.array([pcx, pcy]))
                    if dist_to_phone < min_dist:
                        min_dist = dist_to_phone
                        closest_student_id = track_id
                
                if closest_student_id is not None and min_dist < 260:
                    students[closest_student_id]['is_cheating'] = True
                    students[closest_student_id]['reason'] = "Phone Usage"

            # --- RENDER CUSTOM BOXES & TRIGGER FIREBASE LOGS ---
            for track_id, data in students.items():
                x1, y1, x2, y2 = data['box']
                
                if data['is_cheating']:
                    box_color = (0, 0, 255)  # RED for suspects
                    label = f"Student ID: {track_id} [{data['reason']}]"
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 4)
                    cv2.putText(display_frame, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                    # Trigger alert uploads and local logging
                    alert_key = f"{video_file}_{track_id}_{data['reason']}"
                    if alert_key not in captured_cheaters:
                        current_time = datetime.now()
                        timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
                        
                        # Define local file metadata
                        screenshot_name = f"cheat_{video_file[:-4]}_id_{track_id}_{timestamp_str}.jpg"
                        screenshot_path = os.path.join(screenshot_dir, screenshot_name)
                        
                        # Save screenshot locally with active bounding box lines
                        cv2.imwrite(screenshot_path, display_frame)
                        print(f"⚠️ LOCAL ALERT: {data['reason']} caught for ID {track_id}!")

                        # 🌐 SYNC TO CLOUD FIREBASE DATABASE
                        try:
                            alert_data = {
                                "student_id": int(track_id),
                                "cheat_type": str(data['reason']),
                                "video_source": str(video_file),
                                "timestamp": current_time.isoformat(),
                                "screenshot_path": str(screenshot_path)
                            }
                            # Send data to a collection named 'cheating_alerts'
                            db.collection("cheating_alerts").add(alert_data)
                            print(f"🌐 CLOUD SYNC: Alert details uploaded to Firebase Firestore successfully!")
                        except Exception as e:
                            print(f"❌ Firebase Cloud Sync Failed: {e}")
                            
                        captured_cheaters.add(alert_key)
                else:
                    box_color = (0, 255, 0)  # GREEN for normal behavior
                    label = f"Student ID: {track_id}"
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 4)
                    cv2.putText(display_frame, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Render window loop
        cv2.imshow("Cheating Tracking and Alert System", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(f"Skipped video: {video_file} by user request.")
            break

    cap.release()

cv2.destroyAllWindows()
print("👋 All videos processed successfully.")
