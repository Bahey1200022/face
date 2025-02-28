from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
import io
from PIL import Image
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import uvicorn
app = FastAPI()

# Load known faces
try:
    with open("known_faces.pkl", "rb") as f:
        known_faces = pickle.load(f)
except FileNotFoundError:
    known_faces = {}

# Initialize face analysis model
app_insight = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app_insight.prepare(ctx_id=-1, det_size=(640, 640))

def recognize_face(face_embedding):
    """Compare face embedding to known faces and return best match."""
    best_match = "Unknown"
    best_score = 0.0
    for name, known_embedding in known_faces.items():
        score = cosine_similarity([face_embedding], [known_embedding])[0][0]
        if score > best_score and score > 0.5:
            best_match = name
            best_score = score
    return best_match

@app.post("/predict", include_in_schema=False)
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    frame = np.array(image)

    # Run face recognition
    faces = app_insight.get(frame)
    
    if not faces:  # No faces detected
        return JSONResponse(content={}, status_code=204)  # No Content response

    recognized_names = []
    
    for face in faces:
        embedding = face.embedding
        name = recognize_face(embedding)
        recognized_names.append(name)

        # Draw bounding box
        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Save processed image
    output_path = "processed_image.jpg"
    cv2.imwrite(output_path, frame)
    
    # Return both image and recognized names
    return JSONResponse(content={"name": recognized_names}, status_code=200)



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
