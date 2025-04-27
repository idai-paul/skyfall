"""
Smart City Surveillance System
"""

from flask import Flask, render_template, jsonify, redirect, url_for, request
import os
import time
import json

app = Flask(__name__)

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
        "detected_subjects": []
    },
    {
        "id": "CAM-04",
        "name": "Main Street",
        "status": "online",
        "detected_subjects": ["28F4"]
    },
    {
        "id": "CAM-12",
        "name": "Market Square", 
        "status": "online",
        "detected_subjects": ["19A7"]
    },
    {
        "id": "CAM-07",
        "name": "Bus Stop",
        "status": "online",
        "detected_subjects": []
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
    return render_template('index.html', 
                           tracked_subjects=TRACKED_SUBJECTS,
                           camera_feeds=CAMERA_FEEDS,
                           system_status=SYSTEM_STATUS)

@app.route('/api/subjects')
def get_subjects():
    return jsonify(TRACKED_SUBJECTS)

@app.route('/api/cameras')
def get_cameras():
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

# Route to get placeholder video feeds (for future implementation)
@app.route('/api/video/<camera_id>')
def get_video_feed(camera_id):
    # In a real implementation, this would stream video from the camera
    # For now, just return a status message
    return jsonify({
        "status": "Video feed not available",
        "message": f"Placeholder for camera {camera_id} video feed"
    })

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

if __name__ == '__main__':
    app.run(debug=True, port=8080)
"""
Smart City Surveillance System

A Flask application to display the Smart City Surveillance System with person segmentation.
The application includes placeholder cards for live video feeds and stubs for future integration.
Using HTML and CSS instead of SVG.
"""

from flask import Flask, render_template, jsonify
import os
import time

app = Flask(__name__)

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
    },
    {
        "id": "19A7",
        "confidence": "87%",
        "first_seen": "14:12:55",
        "location": "Market Sq",
        "signature": "Red coat",
        "path": ["Bus Stop", "City Park", "Market"],
        "color": "#00AAFF",
    }
]

# Sample data for camera feeds
CAMERA_FEEDS = [
    {
        "id": "CAM-01",
        "name": "Central Station",
        "status": "online",
        "detected_subjects": []
    },
    {
        "id": "CAM-04",
        "name": "Main Street",
        "status": "online",
        "detected_subjects": ["28F4"]
    },
    {
        "id": "CAM-12",
        "name": "Market Square", 
        "status": "online",
        "detected_subjects": ["19A7"]
    },
    {
        "id": "CAM-07",
        "name": "Bus Stop",
        "status": "online",
        "detected_subjects": []
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
    svg_content = get_svg_content()
    return render_template('index.html', 
                           svg_content=svg_content,
                           tracked_subjects=TRACKED_SUBJECTS,
                           camera_feeds=CAMERA_FEEDS,
                           system_status=SYSTEM_STATUS)

@app.route('/api/subjects')
def get_subjects():
    return jsonify(TRACKED_SUBJECTS)

@app.route('/api/cameras')
def get_cameras():
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

if __name__ == '__main__':
    # Run only on localhost with port 8080
    app.run(host='127.0.0.1', port=8080, debug=True)