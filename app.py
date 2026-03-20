import cv2
import face_recognition
from utils import SecurityLogic

brain = SecurityLogic("data/owner.jpg")
video_capture = cv2.VideoCapture(0)

frame_count = 0;
status_text, color = "SCANNING...", (255, 255, 255)
face_locations = []

while True: 
   ret, frame = video_capture.read()
   if not ret: break

   frame_count += 1

   if frame_count % 5 == 0:
      small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
      rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

      face_locations = face_recognition.face_locations(rgb_small_frame)
      face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

      for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
         status_text, color = brain.analyze_face(frame, (top, right, bottom, left), encoding)

   for (top, right, bottom, left) in face_locations:
      t, r, b, l = top * 4, right * 4, bottom * 4, left * 4
      
      cv2.rectangle(frame, (l, t), (r, b), color, 2)
      cv2.putText(frame, status_text, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
 
   cv2.imshow('AI Lock System', frame)

   if cv2.waitKey(1) & 0xFF == ord('q'):
      break
      
video_capture.release()
cv2.destroyAllWindows()