from ultralytics import YOLO
import supervision as sv
import cv2
from collections import defaultdict, deque
import time

# Load model
model = YOLO("best-yolov11s-datasetv14-95P-84R-91A.pt")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Tracking configuration
TRACK_CONFIRM_FRAMES = 5
REMOVAL_FRAMES = 30  # Remove item if not seen for 30 frames

# Data structures
track_history = defaultdict(lambda: deque(maxlen=TRACK_CONFIRM_FRAMES))
confirmed_tracks = set()
cart = defaultdict(int)
track_last_seen = defaultdict(int)
track_confirmed_class = {}  # Store what class each track was confirmed as

# FPS tracking
fps_start = time.time()
fps_count = 0
current_fps = 0

# Frame counter
frame_count = 0

print("🎥 Starting YOLO Checkout System...")
print("Controls:")
print("  'q' - Quit")
print("  'c' - Clear cart")
print("  'p' - Print cart to console")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    fps_count += 1

    # Calculate FPS every 30 frames
    if fps_count >= 30:
        current_fps = 30 / (time.time() - fps_start)
        fps_start = time.time()
        fps_count = 0

    # Run YOLO tracking
    results = model.track(
        frame,
        conf=0.4,
        imgsz=640,
        persist=True,
        tracker="bytetrack.yaml"
    )[0]

    detections = sv.Detections.from_ultralytics(results)

    # Create annotated frame
    annotated_frame = frame.copy()

    # Handle detections
    if detections.tracker_id is not None:
        active_tracks = set(int(tid) for tid in detections.tracker_id)

        # Update last seen time for active tracks
        for track_id in active_tracks:
            track_last_seen[track_id] = frame_count

        # Process each detection
        for i in range(len(detections)):
            track_id = int(detections.tracker_id[i])
            class_id = int(detections.class_id[i])
            confidence = float(detections.confidence[i])
            class_name = model.names[class_id]

            # Store votes for this track
            track_history[track_id].append(class_name)

            # Check if track should be confirmed
            if track_id not in confirmed_tracks:
                votes = track_history[track_id]
                if len(votes) == TRACK_CONFIRM_FRAMES:
                    # Count votes for each class
                    vote_counts = defaultdict(int)
                    for v in votes:
                        vote_counts[v] += 1
                    
                    # Get majority class
                    majority_class = max(vote_counts, key=vote_counts.get)
                    
                    # Confirm if 4 or more votes for majority class
                    if vote_counts[majority_class] >= 4:
                        confirmed_tracks.add(track_id)
                        track_confirmed_class[track_id] = majority_class
                        cart[majority_class] += 1
                        print(f"🛒 Added to cart: {majority_class} (Track ID: {track_id})")

            # Determine UI state and color
            if track_id in confirmed_tracks:
                state = "CONFIRMED"
                color = (0, 255, 0)  # Green
                display_name = track_confirmed_class.get(track_id, class_name)
            elif len(track_history[track_id]) >= 3:
                state = "VERIFYING"
                color = (0, 255, 255)  # Yellow
                display_name = class_name
            else:
                state = "DETECTING"
                color = (0, 0, 255)  # Red
                display_name = class_name

            # Draw bounding box and label
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{display_name} [{state}]"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Check for tracks that should be removed (not seen for REMOVAL_FRAMES)
        tracks_to_remove = []
        for track_id in list(confirmed_tracks):
            if frame_count - track_last_seen[track_id] > REMOVAL_FRAMES:
                tracks_to_remove.append(track_id)

        # Remove tracks
        for track_id in tracks_to_remove:
            if track_id in track_confirmed_class:
                removed_class = track_confirmed_class[track_id]
                if cart[removed_class] > 0:
                    cart[removed_class] -= 1
                    print(f"🗑️ Removed from cart: {removed_class} (Track ID: {track_id})")
                    if cart[removed_class] == 0:
                        del cart[removed_class]
                del track_confirmed_class[track_id]
            
            confirmed_tracks.discard(track_id)
            if track_id in track_history:
                del track_history[track_id]

        # Clean up stale track history
        stale_tracks = set(track_history.keys()) - active_tracks - confirmed_tracks
        for track_id in stale_tracks:
            del track_history[track_id]

    # Draw cart overlay
    cart_bg_height = 40 + len(cart) * 35
    cv2.rectangle(annotated_frame, (10, 10), (320, cart_bg_height), (0, 0, 0), -1)
    cv2.rectangle(annotated_frame, (10, 10), (320, cart_bg_height), (255, 255, 255), 2)
    
    # Cart header
    cv2.putText(
        annotated_frame,
        "🛒 CART",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Cart items
    y_offset = 35
    if len(cart) == 0:
        y_offset += 35
        cv2.putText(
            annotated_frame,
            "Empty",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1
        )
    else:
        for item, count in cart.items():
            y_offset += 35
            cv2.putText(
                annotated_frame,
                f"{item}: {count}x",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    # Draw FPS counter
    fps_text = f"FPS: {current_fps:.1f}"
    cv2.putText(
        annotated_frame,
        fps_text,
        (frame.shape[1] - 120, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    # Draw instructions
    instructions = "Q:Quit | C:Clear | P:Print"
    cv2.putText(
        annotated_frame,
        instructions,
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # Show frame
    cv2.imshow("YOLO Checkout System", annotated_frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("\n👋 Shutting down...")
        break
    elif key == ord("c"):
        # Clear cart
        cart.clear()
        confirmed_tracks.clear()
        track_confirmed_class.clear()
        track_history.clear()
        print("\n🧹 Cart cleared!")
    elif key == ord("p"):
        # Print cart to console
        print("\n📋 Current Cart:")
        if len(cart) == 0:
            print("  (empty)")
        else:
            total_items = 0
            for item, count in cart.items():
                print(f"  {item}: {count}x")
                total_items += count
            print(f"  Total items: {total_items}")

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Final cart summary
print("\n" + "="*50)
print("📊 Final Cart Summary")
print("="*50)
if len(cart) == 0:
    print("No items in cart")
else:
    total_items = 0
    for item, count in cart.items():
        print(f"{item}: {count}x")
        total_items += count
    print(f"\nTotal items: {total_items}")
print("="*50)