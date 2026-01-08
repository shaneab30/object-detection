from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import base64
import numpy as np
import cv2
import time
from collections import defaultdict
from flask import request

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")
user_carts = defaultdict(lambda: defaultdict(int))
tracked_objects = defaultdict(dict)
CONFIDENCE_THRESHOLD = 0.6
TRACKING_ITERS = 1  # Enable ByteTrack tracking

model = YOLO("best-yolov11s-datasetv18-95P-85R-92A.pt")

@app.route('/')
def index():
    return {"message" : "YOLOV11 Checkout API is running."}

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'data': 'Connected to server'})
    
@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    user_carts.pop(sid, None)
    tracked_objects.pop(sid, None)
    print('Client disconnected')
    
@socketio.on("image")
def handle_image(data):
    sid = request.sid
    user_cart = user_carts[sid]
    user_tracks = tracked_objects[sid]
    
    try:
        img_data = base64.b64decode(data.split(",")[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            emit("error", {"message": "Failed to decode image"})
            return

        start = time.time()
        
        # Enable tracking with YOLO
        results = model.track(
            frame, 
            conf=0.5, 
            imgsz=320,  # Smaller = faster (try 320 for even more speed)
            persist=True, 
            tracker="bytetrack.yaml",
            verbose=False  # Disable verbose logging for speed
        )[0]
        
        end = time.time()
        fps = 1.0 / (end - start)

        detections = []
        current_track_ids = set()

        boxes = results.boxes
        if boxes is not None and boxes.id is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                track_id = int(box.id[0]) if box.id is not None else None

                detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "track_id": track_id
                })

                # Only add to cart if this tracked object hasn't been added before
                if track_id is not None:
                    current_track_ids.add(track_id)
                    
                    if track_id not in user_tracks:
                        user_tracks[track_id] = {'label': label, 'added': False}
                    
                    if not user_tracks[track_id]['added']:
                        user_cart[label] += 1
                        user_tracks[track_id]['added'] = True
                        print(f"Added to cart: {label} (Track ID: {track_id})")

        # Clean up tracks that are no longer visible (optional)
        # Remove tracks not seen for 5 seconds
        tracks_to_remove = []
        for track_id in list(user_tracks.keys()):
            if track_id not in current_track_ids:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove[:10]:  # Remove max 10 old tracks per frame
            del user_tracks[track_id]

        emit("detections", {
            "detections": detections,
            "fps": round(fps, 1),
            "cart": dict(user_cart)
        })

    except Exception as e:
        print(f"Error: {e}")
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    socketio.run(app, port=5000, debug=True, allow_unsafe_werkzeug=True)