import cv2
from ultralytics import YOLO
import pyttsx3

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Variables to store detected objects and bounding boxes
detected_objects = []
bounding_boxes = []

def mouse_callback(event, x, y, flags, param):
    global detected_objects, bounding_boxes
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (box, label) in enumerate(zip(bounding_boxes, detected_objects)):
            x1, y1, x2, y2 = box
            if x1 <= x <= x2 and y1 <= y <= y2:
                print(f"Clicked on: {label}")
                speak(label)
                break

# Start the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", mouse_callback)

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            continue
        
        # Run YOLO detection on the captured frame
        results = model(frame)
        
        # Clear previous annotations
        detected_objects = []
        bounding_boxes = []

        # Annotate the frame with detection results
        for result in results[0].boxes.data:
            x1, y1, x2, y2 = map(int, result[:4])
            label = results[0].names[int(result[5])]
            detected_objects.append(label)
            bounding_boxes.append((x1, y1, x2, y2))
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the live video feed with annotations
        cv2.imshow("Camera", frame)

        # Capture photo on space key press
        if cv2.waitKey(1) & 0xFF == ord(' '):
            print("Photo captured!")

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed")
