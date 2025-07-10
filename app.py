import base64
import os
import shutil
import threading
import time
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
from ultralytics import YOLO

# --- Configuration ---
CONF_THRESHOLD = 0.35
SAVE_COOLDOWN = 5.0  # Seconds
SAVE_DIR = "detected_objects"
STATIC_DIR = "static"  # For serving images like robots
DANGEROUS_CLASSES = ["boulder", "person", "branch"]
CLEAR_CLASS = "clear" # The class name for a safe condition

# --- Configuration for Display Size ---
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

# --- Flask & Socket.IO App Initialization ---
app = Flask(__name__, template_folder='templates', static_folder=STATIC_DIR)
app.config['SECRET_KEY'] = 'your_secret_key_here' # Change this!
socketio = SocketIO(app, async_mode='eventlet')

# --- Global Variables for Camera Control ---
camera_thread = None
camera_active = False

# --- Load YOLOv8 Model ---
try:
    model = YOLO("runs/detect/train/weights/best.pt")
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Create Directories if they don't exist ---
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


# --- Camera Detection Loop (The Core Logic) ---
def run_camera_detection():
    """This function runs in a background thread to handle camera detection."""
    global camera_active
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    last_save_time = {}
    print("Camera thread started. Waiting for 'start_camera' event.")

    while camera_active:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model.predict(source=frame, conf=CONF_THRESHOLD, iou=0.5)

        # Process results
        result = results[0]
        annotated_frame = frame.copy()
        current_time = time.time()
        
        is_dangerous_detected = False
        is_clear_detected = False # <<< NEW: Flag for clear condition
        
        detection_labels = []

        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf.item()
            cls_id = int(box.cls.item())
            label = model.names[cls_id]
            
            label_text = f"{label} (Confidence: {int(conf * 100)}%)"
            detection_labels.append(label_text)

            # <<< MODIFIED: Check for dangerous and clear classes >>>
            if label.lower() in DANGEROUS_CLASSES:
                is_dangerous_detected = True
                box_color = (0, 0, 255)  # Red for danger
            elif label.lower() == CLEAR_CLASS:
                is_clear_detected = True
                box_color = (0, 255, 0)  # Green for clear
            else:
                box_color = (255, 255, 0)  # Cyan for other objects

            display_text = f'{label} {int(conf * 100)}%'
            cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), box_color, 2)
            cv2.putText(annotated_frame, display_text, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            if current_time - last_save_time.get(cls_id, 0) > SAVE_COOLDOWN:
                x1, y1, x2, y2 = xyxy
                cropped_object = frame[y1:y2, x1:x2]
                if cropped_object.size > 0:
                    timestamp = int(time.time() * 1000)
                    filename = os.path.join(SAVE_DIR, f"{label}_{timestamp}.jpg")
                    cv2.imwrite(filename, cropped_object)
                    print(f"Saved: {filename}")
                    last_save_time[cls_id] = current_time

        # <<< NEW: Determine the overall alert status for the frame >>>
        alert_status = 'monitoring' # Default status
        if is_dangerous_detected:
            alert_status = 'danger'
        elif is_clear_detected:
            alert_status = 'clear'

        display_frame = cv2.resize(annotated_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        _, buffer = cv2.imencode('.jpg', display_frame)
        b64_string = base64.b64encode(buffer).decode('utf-8')

        # <<< MODIFIED: Emit the new 'alert_status' to the client >>>
        socketio.emit('image_update', {
            'image': b64_string,
            'count': len(result.boxes),
            'dangerous': is_dangerous_detected, # For the siren
            'detections': detection_labels,
            'alert_status': alert_status # The new alert status
        })
        
        socketio.sleep(0.03)

    cap.release()
    print("Camera thread stopped.")

# --- Flask Routes ---
@app.route('/')
def index():
    """Serve the main HTML file."""
    return render_template('index.html')

@app.route('/detected_objects/<filename>')
def detected_object_image(filename):
    """Serve images from the gallery."""
    return send_from_directory(SAVE_DIR, filename)

@app.route('/static/<filename>')
def static_files(filename):
    """Serve static files like the robot images."""
    return send_from_directory(STATIC_DIR, filename)

# --- Socket.IO Event Handlers ---
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    global camera_active
    camera_active = False
    print('Client disconnected')

@socketio.on('start_camera')
def handle_start_camera():
    """Starts the camera detection thread."""
    global camera_thread, camera_active
    if not camera_active:
        camera_active = True
        camera_thread = socketio.start_background_task(target=run_camera_detection)
        print("Camera detection started by client.")

@socketio.on('stop_camera')
def handle_stop_camera():
    """Stops the camera detection thread."""
    global camera_active
    camera_active = False
    print("Camera detection stopped by client.")

@socketio.on('get_gallery_images')
def handle_get_gallery():
    """Sends the list of saved images to the client."""
    try:
        images = os.listdir(SAVE_DIR)
        image_data = []
        for img_name in sorted(images, reverse=True):
            class_name = img_name.split('_')[0]
            image_data.append({
                'url': f'/detected_objects/{img_name}',
                'class': class_name
            })
        socketio.emit('gallery_content', image_data)
        print(f"Sent {len(image_data)} images to gallery.")
    except Exception as e:
        print(f"Error getting gallery images: {e}")

@socketio.on('delete_all_images')
def handle_delete_all_images():
    """Deletes all images from the gallery directory."""
    try:
        shutil.rmtree(SAVE_DIR)
        os.makedirs(SAVE_DIR, exist_ok=True)
        print("All gallery images deleted.")
        socketio.emit('gallery_deleted', {'message': 'All images have been successfully deleted.'})
    except Exception as e:
        print(f"Error deleting gallery images: {e}")
        socketio.emit('gallery_deleted', {'message': 'Error: Could not delete images.'})

# --- Main Entry Point ---
if __name__ == '__main__':
    print("Starting server on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000)