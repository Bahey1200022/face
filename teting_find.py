import cv2
import pickle
import numpy as np
import threading
from deepface import DeepFace

# Load known face embeddings
try:
    with open("known_faces.pkl", "rb") as f:
        known_faces = pickle.load(f)
except FileNotFoundError:
    known_faces = {}

# Initialize global variables
current_frame = None
face_results = None

def process_frame():
    """Thread function to process frames in the background."""
    global current_frame, face_results

    while True:
        if current_frame is not None:
            frame_resized = cv2.resize(current_frame, (640, 480))  # Resize for speed
            try:
                # Use deepface.find to recognize faces
                face_results = DeepFace.find(
                    img_path="m.jpg",
                    db_path="db",  # Path to your database of known faces
                    model_name="Facenet",  # Face recognition model
                    detector_backend="ssd",  # Use a supported backend
                    enforce_detection=False,
                    silent=True
                )
                print(face_results)
            except Exception as e:
                print(f"Error in face detection: {e}")
                face_results = None

# Start a background thread for face detection
thread = threading.Thread(target=process_frame, daemon=True)
thread.start()

# Open webcam
cap = cv2.VideoCapture(0)
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    if frame_counter % 5 == 0:  # Only process every 5th frame
        current_frame = frame.copy()

    if face_results:
        for result in face_results:
            for _, row in result.iterrows():
                try:
                    # Extract bounding box and identity
                    x, y, w, h = row["source_x"], row["source_y"], row["source_w"], row["source_h"]
                    identity = row["identity"]
                    similarity = row["Facenet_cosine"]

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"{identity} ({similarity:.2f})"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error: {e}")

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()