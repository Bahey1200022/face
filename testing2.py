import cv2
import pickle
import numpy as np
import threading
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

# Load known face embeddings
try:
    with open("known_faces.pkl", "rb") as f:
        known_faces = pickle.load(f)
except FileNotFoundError:
    known_faces = {}

# Load DeepFace model (Facenet)
model = DeepFace.build_model("Facenet")

# Initialize global variables
current_frame = None
face_results = None

def recognize_face(face_embedding):
    """Compare face embedding to known faces and return the best match."""
    best_match = "Unknown"
    best_score = 0.0

    for name, known_embedding in known_faces.items():
        score = cosine_similarity([face_embedding], [known_embedding])[0][0]
        if score > best_score and score > 0.5:  # Threshold (adjust if needed)
            best_match = name
            best_score = score

    return best_match

def process_frame():
    """Thread function to process frames in the background."""
    global current_frame, face_results

    while True:
        if current_frame is not None:
            frame_resized = cv2.resize(current_frame, (640, 480))  # Resize for speed
            try:
                face_results = DeepFace.represent(frame_resized, model_name="Facenet", detector_backend="ssd", enforce_detection=False)
            except:
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
        for face in face_results:
            try:
                face_embedding = face["embedding"]
                name = recognize_face(face_embedding)

                # Extract bounding box correctly
                facial_area = face["facial_area"]
                x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error: {e}")

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
