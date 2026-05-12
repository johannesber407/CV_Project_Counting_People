from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo26n.pt")

# Open input video
cap = cv2.VideoCapture("C:\\Users\\jojob\\Nextcloud\\Erasmus\\Kurse\\computer vision\\CV_Project_Counting_People-1\\examples\\example.mp4")

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# # Output video writer
# out = cv2.VideoWriter(
#     "output.mp4",
#     cv2.VideoWriter_fourcc(*"mp4v"),
#     fps,
#     (width, height)
# )

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run detection
    results = model(frame)
    print(results[0].boxes)

    # Process detections
    for box in results[0].boxes:
        cls = int(box.cls[0])

        # COCO class 0 = person
        if cls == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"Person {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Save frame
#    out.write(frame)

    # Optional live preview
    cv2.imshow("Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
##out.release()
cv2.destroyAllWindows()