import cv2
import os

name = input("Enter name for this face: ")
save_dir = "authorized_faces"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
print ("Press 's' to save your photo, press 'q' to quit")

while True: 
    ret, frame = cap.read()
    if not ret: 
        break

    cv2.imshow("Add face - press 's' to save", frame)

    key = cv2.waitKeyEx(1) & 0xFF

    if key == ord('s'):
        path = os.path.join(save_dir, f"{name}.jpg")
        cv2.imwrite(path ,frame)
        print(f"Saved to {path}")
        break
    
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()