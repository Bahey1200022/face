import cv2
import time
import os
import numpy as np
import pickle  # To save/load face embeddings
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity  # For face matching
from mail import send_email

# Create screenshots directory if it doesn't exist
if not os.path.exists('screenshots'):
    os.makedirs('screenshots')




# Load known faces (if available)
try:
    with open("known_faces.pkl", "rb") as f:
        known_faces = pickle.load(f)
    print("Loaded known faces.")
except FileNotFoundError:
    known_faces = {}  # Dictionary to store embeddings ({"name": embedding})


# ... existing camera setup code ...

last_capture_time = time.time()

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
# Camera settings
IP = "192.168.0.105"
USERNAME = "admin"
PASSWORD = "moataz2019"
RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{IP}:554/cam/realmonitor?channel=1&subtype=0"


##  model init 
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))


# Initialize video capture with error handling
# cap = cv2.VideoCapture(RTSP_URL)    ip camera
cap=cv2.VideoCapture(0)  # webcam

# Check if camera opened successfully
if not cap.isOpened():
    print(f"Error: Could not connect to camera at {RTSP_URL}")
    exit()

print("Successfully connected to camera")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame")
        break
    
    # Display the frame
    cv2.imshow("IP Camera Stream", frame)
    
    # Capture screenshot every 2 seconds
    current_time = time.time()
    if current_time - last_capture_time >= 2.0:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        last_capture_time = current_time
        
        
           # Run face recognition on the screenshot
        faces = app.get(frame)
        name = "Unknown"  # Default to Unknown in case no faces are detected
        for face in faces:
            box = face.bbox.astype(int)
            embedding = face.embedding

            # Recognize face
            name = recognize_face(embedding)

            # Draw bounding box and label
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)
            filename = f"screenshots/capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")    
            alert_type = "Unknown Person" if name == "Unknown" else "Known Person"
            send_email(
            f"👤 {alert_type} Detected",
            f"Person detected: {name}",
            attachment_path=filename,
            sender="bahy2002@gmail.com",
            password="utqx qvvh ebea vuxx",
            recipients=["M.ahmed201017@gmail.com"])          
        
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()