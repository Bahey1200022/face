from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import io
import base64
from PIL import Image
from deepface import DeepFace
import pickle
import uvicorn
import logging
from fastapi.middleware.cors import CORSMiddleware
logging.basicConfig(
    filename="app2.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logging.info("Starting application...")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load known faces
try:
    with open("known_faces.pkl", "rb") as f:
        known_faces = pickle.load(f)
except FileNotFoundError:
    known_faces = {}

# Load DeepFace model
logging.info("Loading DeepFace model...")
DeepFace.build_model("Facenet")
logging.info("DeepFace model loaded successfully.")

def recognize_face(face_embedding, threshold=0.6):
    best_match = "Unknown"
    best_score = 0.0
    for name, known_embedding in known_faces.items():
        score = np.dot(face_embedding, known_embedding) / (np.linalg.norm(face_embedding) * np.linalg.norm(known_embedding))
        if score > best_score and score > threshold:
            best_match = name
            best_score = score
    return best_match

# @app.post("/v1/chat/completions")
# async def chat_completions(request: dict):
#     try:
#         logging.info(f"Received request: {request}")
#         if "model" not in request or "messages" not in request:
#             raise HTTPException(status_code=400, detail="Invalid OpenAI API format")
        
#         base64_image = None
#         for msg in request["messages"]:
#             if msg["role"] == "user":
#                 content = msg["content"]
#                 if isinstance(content, list):
#                     for item in content:
#                         if isinstance(item, dict) and item.get("type") == "image_url":
#                             image_url = item["image_url"]["url"]
#                             if image_url.startswith("data:image/"):
#                                 base64_image = image_url.split(",")[1]
#                                 break
        
#         recognized_names = []
#         processed_image_base64 = ""

#         if base64_image:
#             try:
#                 image_data = base64.b64decode(base64_image)
#                 image = Image.open(io.BytesIO(image_data)).convert("RGB")
#                 frame = np.array(image)
                
#                 analysis = DeepFace.represent(img_path=frame, model_name="Facenet")
#                 if analysis:
#                     for face in analysis:
#                         embedding = face["embedding"]
#                         name = recognize_face(embedding)
#                         recognized_names.append(name)
#                     # Draw bounding boxes
#                     for face in analysis:
#                         box = face.bbox.astype(int)
#                         cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
#                         cv2.putText(frame, name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
#                                     0.6, (0, 255, 0), 2, cv2.LINE_AA)
#                         cv2.imwrite("image.jpg", frame)

#                     # Convert processed image to Base64
#                     _, buffer = cv2.imencode(".jpg", frame)
#                     processed_image_base64 = base64.b64encode(buffer).decode("utf-8")
#                 else:
#                     recognized_names = ["No face detected"]
                        

#                 response = {
#                     "id": "chatcmpl-123",
#                     "object": "chat.completion",
#                     "created": 1710000000,
#                     "model": request["model"],
#                     "choices": [
#                         {
#                             "index": 0,
#                             "message": {
#                                 "role": "assistant",
#                                 "content": {
#                                     "text": f"Recognized faces: {', '.join(recognized_names)}" if recognized_names else "No faces detected.",
#                                     "image": f"data:image/jpeg;base64,{processed_image_base64}"  # Embed Base64 image

                                    
#                                 }
#                             },
#                             "finish_reason": "stop"
#                         }
#                     ]
#                 }
#                 return JSONResponse(content=response, status_code=200)

#             except Exception as img_err:
#                 raise HTTPException(status_code=400, detail=f"Invalid image format: {str(img_err)}")

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    try:
        logging.info(f"Received request: {request}")
        if "model" not in request or "messages" not in request:
            raise HTTPException(status_code=400, detail="Invalid OpenAI API format")

        base64_image = None
        for msg in request["messages"]:
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            image_url = item["image_url"]["url"]
                            if image_url.startswith("data:image/"):
                                base64_image = image_url.split(",")[1]
                                break

        recognized_names = []
        processed_image_base64 = ""

        if base64_image:
            try:
                # Decode image
                image_data = base64.b64decode(base64_image)
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                frame = np.array(image)

                # Detect faces & get embeddings
                analysis = DeepFace.represent(img_path=frame, model_name="Facenet", detector_backend="mtcnn")

                if analysis:
                    for face in analysis:
                        embedding = face["embedding"]
                        name = recognize_face(embedding)
                        recognized_names.append(name)

                    # Convert processed image to Base64
                    _, buffer = cv2.imencode(".jpg", frame)
                    processed_image_base64 = base64.b64encode(buffer).decode("utf-8")
                else:
                    recognized_names = ["No face detected"]

                response = {
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1710000000,
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": {
                                    "text": f"Recognized faces: {', '.join(recognized_names)}" if recognized_names else "No faces detected.",
                                    "image": f"data:image/jpeg;base64,{processed_image_base64}"  # Embed Base64 image
                                }
                            },
                            "finish_reason": "stop"
                        }
                    ]
                }
                return JSONResponse(content=response, status_code=200)

            except Exception as img_err:
                raise HTTPException(status_code=400, detail=f"Invalid image format: {str(img_err)}")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
@app.post("/api/add_face")
async def add_face(name: str = Form(...), image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        frame = np.array(image)
        
        analysis = DeepFace.represent(img_path=frame, model_name="Facenet",detector_backend="ssd")
        if not analysis:
            raise HTTPException(status_code=400, detail="No face detected")
        
        embedding = analysis[0]["embedding"]
        known_faces[name] = embedding
        with open("known_faces.pkl", "wb") as f:
            pickle.dump(known_faces, f)
        
        return {"message": f"Face for '{name}' added successfully!"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
