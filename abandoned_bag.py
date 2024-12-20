import cv2
from ultralytics import YOLO
import time
import os
import torch

# Load the YOLOv8 model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = YOLO("yolov8n.pt")  # Use an appropriate YOLOv8 model
model.to(device)

# Start video capture
cap = cv2.VideoCapture(0)

# Parameters
movement_tolerance = 10  # Movement tolerance for stationary check, mine is 10 
stationary_duration = 5  # Time in seconds to classify as risky, mine is 5 seconds

# Object tracking data structure
object_tracker = {}

# Check if video capture is successful
if not cap.isOpened():
    print("Error: Unable to open video stream.")
    exit()

# Create a directory to save risky frames
save_dir = "risky_frames"    #create a directory in the working folder 
os.makedirs(save_dir, exist_ok=True)

image_saved = False
#risky_objects = {}
# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame, device=device)

    # Extract detected objects
    current_objects = []
    for box in results[0].boxes:
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].cpu().numpy())  # Move to CPU before converting to NumPy
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        class_id = int(box.cls.cpu().numpy())  # Ensure the class ID tensor is on CPU
        label = model.names[class_id]

        if label in ["backpack", "suitcase", "handbag"]:
            current_objects.append({"center": (center_x, center_y), "bbox": (x_min, y_min, x_max, y_max), "label": label})

    # Current time
    current_time = time.time()

    # Update object tracker
    for obj in current_objects:
        obj_id = None
        min_distance = float('inf')

        # Match objects with existing tracker entries
        for existing_id, data in object_tracker.items():
            last_position = data["last_position"]
            if last_position:
                distance = ((obj["center"][0] - last_position[0]) ** 2 + (obj["center"][1] - last_position[1]) ** 2) ** 0.5
                if distance < movement_tolerance and distance < min_distance:
                    obj_id = existing_id
                    min_distance = distance

        # Assign new ID if no match found
        if obj_id is None:
            obj_id = len(object_tracker) + 1
            object_tracker[obj_id] = {
                "last_position": obj["center"],
                "first_seen": current_time,
                "last_seen": current_time,
                "stationary_time": 0,
                "label": obj["label"],
                #"bbox": obj["bbox"]
            }
        else:
            # Update existing object
            data = object_tracker[obj_id]
            data["last_position"] = obj["center"]
            data["last_seen"] = current_time
            data["stationary_time"] = current_time - data["first_seen"]

    # Check stationary objects and classify
    for obj_id, data in list(object_tracker.items()):
        if (current_time - data["last_seen"]) > stationary_duration:
            # Remove objects not seen for too long
            del object_tracker[obj_id]
            continue

        # Determine risky or normal status
        status = "Risky" if data["stationary_time"] >= stationary_duration else "Normal"
        color = (0, 0, 255) if status == "Risky" else (0, 255, 0)

        # Save the first risky frame only (the moment it detects bag as risky, it saves the image for further use)
        if status == "Risky" and not data.get("image_saved", False):
            file_name = os.path.join(save_dir, f"risky_object_{obj_id}_{int(current_time)}.jpg")
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, f"ID: {obj_id} | {status}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.imwrite(file_name, frame)
            print(f"Saved Risky frame: {file_name}")
            object_tracker[obj_id]["image_saved"] = True  # Mark as saved

           # Set the flag to True to prevent further image saving
            image_saved = True

        # Find matching object in current_objects
        matching_obj = next((obj for obj in current_objects if obj["center"] == data["last_position"]), None)
        if matching_obj:
            x_min, y_min, x_max, y_max = matching_obj["bbox"]
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, f"ID: {obj_id} | {status}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow("Object Tracking", frame)

    # Break loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()