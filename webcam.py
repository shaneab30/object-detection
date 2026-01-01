from ultralytics import YOLO
import supervision as sv
import cv2
from collections import defaultdict, deque

model = YOLO("best-yolov11s-datasetv15-89P-87R-92A.pt")

cap = cv2.VideoCapture(0)

box_annotator = sv.BoxAnnotator()

TRACK_CONFIRM_FRAMES = 5
track_history = defaultdict(lambda: deque(maxlen=TRACK_CONFIRM_FRAMES))
confirmed_tracks = set()
cart = defaultdict(int)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        conf=0.4,
        imgsz=640,
        persist=True,
        tracker="bytetrack.yaml"
    )[0]

    detections = sv.Detections.from_ultralytics(results)

    if detections.tracker_id is None:
        cv2.imshow("YOLO Checkout", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    annotated_frame = frame.copy()

    for i in range(len(detections)):
        track_id = int(detections.tracker_id[i])
        class_id = int(detections.class_id[i])
        confidence = float(detections.confidence[i])
        class_name = model.names[class_id]

        # store votes
        track_history[track_id].append(class_name)

        # check confirmation
        if track_id not in confirmed_tracks:
            votes = track_history[track_id]
            if len(votes) == TRACK_CONFIRM_FRAMES:
                if votes.count(class_name) >= 4:
                    confirmed_tracks.add(track_id)
                    cart[class_name] += 1
                    print(f"Added to cart: {class_name}")

        # UI state
        if track_id in confirmed_tracks:
            state = "CONFIRMED"
            color = (0, 255, 0)
        elif len(track_history[track_id]) >= 3:
            state = "VERIFYING"
            color = (0, 255, 255)
        else:
            state = "DETECTING"
            color = (0, 0, 255)

        # draw bbox + label
        x1, y1, x2, y2 = map(int, detections.xyxy[i])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated_frame,
            f"{class_name} [{state}]",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        
        # Draw cart header
        cv2.putText(
            annotated_frame,
            "CART",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # Draw cart contents (UI)
        y_offset = 60

        for key, value in cart.items():
            cv2.putText(
                annotated_frame,
                f"{key}: {value}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            y_offset += 30


    cv2.imshow("YOLO Checkout", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()