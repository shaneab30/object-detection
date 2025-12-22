from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np

model = YOLO("best-yolov11s-datasetv6.pt")

cap = cv2.VideoCapture(0)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5, imgsz=640)[0]

    # Convert Ultralytics result to Supervision Detections
    detections = sv.Detections.from_ultralytics(results)

    # Get highest confidence detection (NOT accuracy)
    max_conf = detections.confidence.max() if len(detections) > 0 else 0.0

    # Annotate
    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )

    # Display confidence (not accuracy)
    cv2.putText(
        annotated_frame,
        f"Max confidence: {max_conf:.2%}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLO Real-time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
