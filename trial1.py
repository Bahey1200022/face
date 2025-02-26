import cv2
import face_recognition
import pickle
import os

# Database file to store face encodings
DB_FILE = "face_db.pkl"

# Load existing database or create a new one
if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as db_file:
        face_db = pickle.load(db_file)
else:
    face_db = {}

def save_database():
    """Save the face database to a file."""
    with open(DB_FILE, "wb") as db_file:
        pickle.dump(face_db, db_file)

def register_face(frame):
    """Capture a face, encode it, and save it with a given name."""
    name = input("Enter the name for the new face: ").strip()
    if not name:
        print("Invalid name. Registration cancelled.")
        return
    
    face_locations = face_recognition.face_locations(frame)
    if not face_locations:
        print("No face detected. Please try again.")
        return

    face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
    face_db[name] = face_encoding
    save_database()
    print(f"Face registered for {name}.")

def recognize_faces(frame):
    """Recognize faces in the current frame using stored face encodings."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(list(face_db.values()), face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = list(face_db.keys())[match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame
