# app.py
from flask import Flask, render_template, Response, jsonify, request
import cv2
import time
import numpy as np
import threading
import queue
import os
from datetime import datetime
import json
from person_tracker import PersonTracker

app = Flask(__name__)

# Configuration
VIDEO_SOURCES = {
    'A1': {'name': 'Entrance A', 'source': 0},  # 0 = default webcam
    'A2': {'name': 'Lobby', 'source': 'static/demo_videos/lobby.mp4'},
    'B1': {'name': 'Hallway A', 'source': 'static/demo_videos/hallway_a.mp4'},
    'B2': {'name': 'Hallway B', 'source': 'static/demo_videos/hallway_b.mp4'},
    # Add more camera sources as needed
}

# For demonstration, you can use video files or webcams
# In production, you would connect to IP cameras or RTSP streams

# Global state
camera_streams = {}
frame_queues = {}
person_tracker = PersonTracker()
person_clusters = {}
detections_lock = threading.Lock()

# Initialize tracking system
def init_tracking_system():
    """Initialize the person tracking system"""
    for camera_id, camera_info in VIDEO_SOURCES.items():
        frame_queues[camera_id] = queue.Queue(maxsize=10)
        source = camera_info['source']
        
        # Start a thread for each camera
        t = threading.Thread(target=process_camera_feed, 
                            args=(camera_id, source))
        t.daemon = True
        t.start()
    
    # Start the person tracking analysis in a separate thread
    t = threading.Thread(target=analyze_person_tracking)
    t.daemon = True
    t.start()

def process_camera_feed(camera_id, source):
    """Process video from a camera and detect people"""
    # Open the video source
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)  # Webcam
    else:
        cap = cv2.VideoCapture(source)  # Video file
        
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Store the video capture object
    camera_streams[camera_id] = cap
    
    # Process frames
    while True:
        success, frame = cap.read()
        if not success:
            # If it's a video file, loop it
            if not isinstance(source, int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print(f"Error: Failed to get frame from {camera_id}")
                break
        
        # Detect people in the frame
        processed_frame, detections = detect_people(frame, camera_id)
        
        # Add detections to the tracking system
        if detections:
            with detections_lock:
                for detection in detections:
                    person_tracker.add_detection(camera_id, detection)
        
        # Put the processed frame in the queue for streaming
        if not frame_queues[camera_id].full():
            frame_queues[camera_id].put(processed_frame)
        
        # Simulate real-time processing rate
        time.sleep(0.03)  # ~30 fps
    
    cap.release()

def detect_people(frame, camera_id):
    """
    Detect people in a frame using a person detection model
    For this example, we'll simulate detections
    """
    # In a real implementation, you would use a model like YOLO, SSD, or Faster R-CNN
    # For this example, we'll just simulate detections with random boxes
    
    processed_frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Simulate 0-3 person detections
    num_detections = np.random.randint(0, 4)
    detections = []
    
    for i in range(num_detections):
        # Generate a random bounding box
        x = np.random.randint(0, width - 100)
        y = np.random.randint(0, height - 200)
        w = np.random.randint(50, 100)
        h = np.random.randint(100, 200)
        
        # Draw the bounding box
        color = (0, 255, 0)  # Green box
        cv2.rectangle(processed_frame, (x, y), (x+w, y+h), color, 2)
        
        # Create a detection object with features
        detection = {
            'bbox': [x, y, x+w, y+h],
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'features': extract_features(frame, [x, y, x+w, y+h]),
            'confidence': np.random.randint(70, 99)
        }
        
        detections.append(detection)
        
        # Label the detection
        cv2.putText(processed_frame, f"Person {i+1}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return processed_frame, detections

def extract_features(frame, bbox):
    """
    Extract features from a person detection
    This would use a CNN to extract appearance features in a real implementation
    """
    # Simulate feature extraction
    # In a real implementation, you would:
    # 1. Crop the person from the frame using the bbox
    # 2. Resize to a standard size
    # 3. Pass through a feature extractor network (ResNet, etc.)
    # 4. Return the feature vector
    
    # For demonstration, return random features
    return np.random.rand(128).tolist()  # 128-dimensional feature vector

def analyze_person_tracking():
    """Periodically analyze all detections and create person clusters"""
    while True:
        with detections_lock:
            # Run the clustering algorithm
            clusters = person_tracker.update_clusters()
            
            # Update the global clusters
            global person_clusters
            person_clusters = clusters
        
        # Run analysis every second
        time.sleep(1)

def generate_frames(camera_id):
    """Generate frames for video streaming"""
    while True:
        if camera_id in frame_queues and not frame_queues[camera_id].empty():
            frame = frame_queues[camera_id].get()
            
            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in the correct format for a multipart response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # If no frame is available, provide an empty image
            blank_image = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank_image, f"Camera {camera_id} - No Signal", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank_image)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Add a small delay
        time.sleep(0.03)

# Routes
@app.route('/')
def index():
    """Render the main application page"""
    return render_template('index.html', cameras=VIDEO_SOURCES)

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    """Route for streaming video from a specific camera"""
    if camera_id in frame_queues:
        return Response(generate_frames(camera_id),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Camera not found", 404

@app.route('/api/cameras')
def get_cameras():
    """API endpoint to get all camera information"""
    return jsonify(VIDEO_SOURCES)

@app.route('/api/clusters')
def get_clusters():
    """API endpoint to get all person clusters"""
    return jsonify(person_clusters)

@app.route('/api/cluster/<cluster_id>')
def get_cluster(cluster_id):
    """API endpoint to get details about a specific person cluster"""
    if cluster_id in person_clusters:
        return jsonify(person_clusters[cluster_id])
    else:
        return "Cluster not found", 404

@app.route('/api/camera_detections/<camera_id>')
def get_camera_detections(camera_id):
    """API endpoint to get all detections from a specific camera"""
    detections = person_tracker.get_camera_detections(camera_id)
    return jsonify(detections)

if __name__ == '__main__':
    # Make sure the static directory exists
    os.makedirs('static/demo_videos', exist_ok=True)
    
    # Initialize the tracking system
    init_tracking_system()
    
    # Run the Flask app
    app.run(debug=True, threaded=True, host='0.0.0.0')