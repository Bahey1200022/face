import cv2
from trial1 import *
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = recognize_faces(frame)
    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        register_face(frame)
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
