import cv2
from deepface import DeepFace
import os
from datetime import datetime

cap = cv2.VideoCapture(0)
INCIDENTS_DIR = "incidents"
frame_count = 0
ANALYZE_EVERY = 10
last_result = None

last_saved = 0
SAVE_COOLDOWN = 10

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Webcam opened. Press 'q' to quit.")

AUTHORIZED_DIR = "authorized_faces"
SUSPICIOUS_EMOTIONS= ['angry', 'fear', 'disgust']

os.makedirs(INCIDENTS_DIR, exist_ok=True)

def save_incident(frame, label): 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{INCIDENTS_DIR}/{timestamp}_{label}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Incident saved: {filename}")

def is_authorized(frame):
    try: 
        results = DeepFace.find(
            img_path=frame,
            db_path=AUTHORIZED_DIR,
            enforce_detection=False,
            silent=True
        )
        if len(results) > 0 and len(results[0]) >0:
            return True
        return False
    except Exception:
        return False


while True: 
    ret, frame = cap.read()

    if not ret: 
        print("Error: Could not read frame")
        break
    
    frame_count += 1

    if frame_count % ANALYZE_EVERY == 0:
        try: 
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend="mtcnn")
            last_result = result
            emotion = result[0]['dominant_emotion']
            region = result[0]['region']

            x, y, w, h = region['x'], region['y'], region['w'], region['h']

            authorized = is_authorized(frame)

            if authorized: 
                if emotion in SUSPICIOUS_EMOTIONS:
                    color = (0, 165, 255)
                    label = f"AUTHORIZED - SUSPICIOUS: {emotion}"
                else: 
                    color = (0, 255, 0)
                    label = f"AUTHORIZED: {emotion}"
            else: 
                color = (0, 0, 255)
                label = "INTRUDER DETECTED"
                save_incident(frame, label)
                
                now = datetime.now().timestamp()
                if now - last_saved > SAVE_COOLDOWN:
                    save_incident(frame, "INTRDUER")
                    last_saved = now
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
        except Exception as e: 
            print(f"Detection error: {e}")

        if last_result: 
            pass

        cv2.imshow("Deepface - Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()