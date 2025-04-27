"""
Smart City Surveillance System
"""

from flask import Flask, render_template, jsonify, redirect, url_for, request, send_file
import os
import time
import json
import threading
from stream_manager import StreamManager

app = Flask(__name__)

# Initialize the stream manager
stream_manager = StreamManager()

# Add the 4 YouTube live feeds
stream_manager.add_stream("CAM-01", "t4Hl35oF7Dg", "Central Station")
stream_manager.add_stream("CAM-04", "tCubYQP1auE", "Main Street")
stream_manager.add_stream("CAM-12", "cE0dEMM-e0Q", "Market Square")
stream_manager.add_stream("CAM-07", "A1-erGqmWDs", "Bus Stop")

# Start the stream manager in a background thread
stream_thread = threading.Thread(target=stream_manager.start)
stream_thread.daemon = True
stream_thread.start()

# Sample data for the tracked subjects
TRACKED_SUBJECTS = [
    {
        "id": "28F4",
        "confidence": "92%",
        "first_seen": "14:17:22",
        "location": "Main Street",
        "signature": "Black jacket",
        "path": ["Central St", "Main St"],
        "color": "#FF0000",
        "features": ["Black jacket", "Blue cap", "Glasses", "Beard"],
        "age_range": "35-45",
        "gender": "Male",
        "height": "5'10\" (178cm)",
        "recognition_status": "Confirmed",
        "total_appearances": 5
    },
    {
        "id": "19A7",
        "confidence": "87%",
        "first_seen": "14:12:55",
        "location": "Market Sq",
        "signature": "Red coat",
        "path": ["Bus Stop", "City Park", "Market"],
        "color": "#00AAFF",
        "features": ["Red coat", "Backpack", "Hat", "Boots"],
        "age_range": "25-35",
        "gender": "Female",
        "height": "5'6\" (168cm)",
        "recognition_status": "Tentative",
        "total_appearances": 3
    }
]

# Sample data for camera feeds
CAMERA_FEEDS = [
    {
        "id": "CAM-01",
        "name": "Central Station",
        "status": "online",
        "detected_subjects": [],
        "video_id": "t4Hl35oF7Dg"
    },
    {
        "id": "CAM-04",
        "name": "Main Street",
        "status": "online",
        "detected_subjects": ["28F4"],
        "video_id": "tCubYQP1auE"
    },
    {
        "id": "CAM-12",
        "name": "Market Square", 
        "status": "online",
        "detected_subjects": ["19A7"],
        "video_id": "cE0dEMM-e0Q"
    },
    {
        "id": "CAM-07",
        "name": "Bus Stop",
        "status": "online",
        "detected_subjects": [],
        "video_id": "A1-erGqmWDs"
    }
]

# Sample data for system status
SYSTEM_STATUS = {
    "cameras": "24/24",
    "tracked": 12,
    "algorithm": "FaceTrack v3"
}

# Routes
@app.route('/')
def index():
    # Update camera feeds with stream status
    for feed in CAMERA_FEEDS:
        feed["stream_status"] = stream_manager.get_stream_status(feed["id"])
    
    return render_template('index.html', 
                           tracked_subjects=TRACKED_SUBJECTS,
                           camera_feeds=CAMERA_FEEDS,
                           system_status=SYSTEM_STATUS)

@app.route('/tracked-people')
def tracked_people():
    return render_template('tracked_people.html',
                           tracked_subjects=TRACKED_SUBJECTS,
                           system_status=SYSTEM_STATUS)

@app.route('/api/subjects')
def get_subjects():
    return jsonify(TRACKED_SUBJECTS)

@app.route('/api/cameras')
def get_cameras():
    # Update camera feeds with stream status
    for feed in CAMERA_FEEDS:
        feed["stream_status"] = stream_manager.get_stream_status(feed["id"])
    
    return jsonify(CAMERA_FEEDS)

@app.route('/api/system')
def get_system_status():
    return jsonify(SYSTEM_STATUS)

@app.route('/api/subject/<subject_id>')
def get_subject(subject_id):
    for subject in TRACKED_SUBJECTS:
        if subject["id"] == subject_id:
            return jsonify(subject)
    return jsonify({"error": "Subject not found"}), 404

# Route to get the latest video clip for a camera
@app.route('/api/video/<camera_id>')
def get_video_feed(camera_id):
    # Get the latest clip for this camera
    latest_clip = stream_manager.get_latest_clip(camera_id)
    
    if latest_clip and os.path.exists(latest_clip):
        # Return the video file
        return send_file(latest_clip, mimetype='video/mp4')
    else:
        # Return a status message if no clip is available
        return jsonify({
            "status": "Video feed not available",
            "message": f"No clip available for camera {camera_id}"
        }), 404

# Route to handle subject profile updates
@app.route('/api/subject/<subject_id>/update', methods=['POST'])
def update_subject(subject_id):
    if request.method == 'POST':
        # This would update the subject in a real implementation
        # For now, just return success
        return jsonify({
            "status": "success",
            "message": f"Subject {subject_id} updated (placeholder)"
        })

# Cleanup function to stop the stream manager when the app is shutting down
@app.teardown_appcontext
def cleanup(exception=None):
    if exception:
        # Log the exception
        app.logger.error(f"Error during cleanup: {exception}")
    
    # Stop the stream manager
    stream_manager.stop()

if __name__ == '__main__':
    app.run(debug=True, port=8080)