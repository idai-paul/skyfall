"""
Updated Flask application for Skyfall project

This module integrates stream_manager and person_processor to create
a complete backend for the Skyfall surveillance system.

Fixes include:
1. Signal handling issues in threaded code
2. Preventing premature service shutdown
3. Adding service monitoring and recovery
"""

from flask import Flask, render_template, jsonify, redirect, url_for, request, send_file, Response
import os
import time
import json
import threading
import logging
import atexit
from datetime import datetime
from stream_manager import StreamManager
from person_processor import PersonProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global flag to track if services are running
services_running = True

# Initialize the stream manager and person processor
stream_manager = StreamManager()
person_processor = PersonProcessor()

# Focus only on Central Station as requested
stream_manager.add_stream("CAM-01", "t4Hl35oF7Dg", "Central Station")

# Start stream manager
try:
    # The StreamManager will create internal threads and pass no_signals=True
    # to the StreamClipper instances to prevent signal handling issues
    stream_manager.start()
    logger.info("Stream manager started successfully")
except Exception as e:
    logger.error(f"Error starting stream manager: {e}")

# Start person processor
try:
    # If person_processor also has signal issues, modify it similar to StreamManager
    processor_thread = threading.Thread(target=person_processor.start)
    processor_thread.daemon = True
    processor_thread.start()
    logger.info("Person processor started successfully")
    # Add a small delay to ensure thread starts properly
    time.sleep(1)
except Exception as e:
    logger.error(f"Error starting person processor: {e}")

# Thread to monitor for new clips and add them to the person processor
def clip_monitor_thread():
    last_processed = {}  # stream_id -> last processed clip
    
    while services_running:  # Use global flag
        try:
            # Check each stream for new clips
            for stream_id in stream_manager.streams:
                latest_clip = stream_manager.get_latest_clip(stream_id)
                
                # Skip if no clip or already processed
                if not latest_clip or latest_clip == last_processed.get(stream_id):
                    continue
                
                # Add to person processor
                person_processor.add_clip(latest_clip, stream_id)
                
                # Update last processed
                last_processed[stream_id] = latest_clip
                logger.info(f"Added clip to queue: {os.path.basename(latest_clip)}")
            
            # Sleep for a bit
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Error in clip monitor: {e}")
            time.sleep(10)  # Longer delay on error

# Start the clip monitor thread
monitor_thread = threading.Thread(target=clip_monitor_thread)
monitor_thread.daemon = True
monitor_thread.start()

# Add a keep-alive thread to prevent services from shutting down
def keep_alive_thread():
    while services_running:
        try:
            # Check if services are still running and restart if needed
            if not stream_manager.is_running:
                logger.warning("Stream manager stopped. Attempting to restart...")
                stream_manager.start()
            
            # Log status periodically (every 30 seconds)
            if int(time.time()) % 30 == 0:
                logger.info("Services keep-alive check - Stream manager running")
            
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error in keep-alive thread: {e}")
            time.sleep(5)

# Start the keep-alive thread
keep_alive_thread = threading.Thread(target=keep_alive_thread)
keep_alive_thread.daemon = True
keep_alive_thread.start()

# Sample data for system status
SYSTEM_STATUS = {
    "cameras": "24/24",
    "tracked": 12,
    "algorithm": "FaceTrack v3"
}

# Routes
@app.route('/')
def index():
    # Get camera feeds with stream status and latest clip info
    camera_feeds = []
    for stream_id, stream_info in stream_manager.streams.items():
        camera_feed = {
            "id": stream_id,
            "name": stream_info.name,
            "status": stream_manager.get_stream_status(stream_id),
            "detected_subjects": [],  # Will be populated from person profiles
            "video_id": stream_info.video_id,
            "latest_clip": stream_manager.get_latest_clip(stream_id),
            "has_clip": False
        }
        
        # Check if clip exists
        if camera_feed["latest_clip"] and os.path.exists(camera_feed["latest_clip"]):
            camera_feed["has_clip"] = True
            
            # Get person profiles for this clip
            profiles = person_processor.get_person_profiles(camera_feed["latest_clip"])
            for person_id in profiles:
                camera_feed["detected_subjects"].append(person_id)
        
        camera_feeds.append(camera_feed)
    
    # Get detected persons for display in person segmentation
    detected_persons = []
    gait_results = []
    
    # Only process Central Station (CAM-01)
    for camera_feed in camera_feeds:
        if camera_feed["id"] == "CAM-01" and camera_feed["latest_clip"]:
            clip_path = camera_feed["latest_clip"]
            clip_name = os.path.splitext(os.path.basename(clip_path))[0]
            #profiles = person_processor.get_person_profiles(clip_path)

            profile_dirs = sorted([d for d in os.listdir("profiles") 
                                 if os.path.isdir(os.path.join("profiles", d))], 
                                 reverse=True)  # newest first
            
            most_recent_profiles = None
            most_recent_clip_name = None
            # Find the most recent directory with completed analysis
            for dir_name in profile_dirs:
                claude_analysis_path = os.path.join("profiles", dir_name, "claude_analysis.json")
                if os.path.exists(claude_analysis_path):
                    # Check if file contains actual profiles
                    with open(claude_analysis_path, 'r') as f:
                        profiles_data = json.load(f)
                        if profiles_data:  # If not empty
                            most_recent_profiles = profiles_data
                            most_recent_clip_name = dir_name
                            logger.info(f"Found most recent completed profiles: {most_recent_clip_name}")
                            break

            # Use the most recent clip name if found, otherwise use the current clip
            if most_recent_clip_name:
                profiles = most_recent_profiles
                logger.info(f"Using profiles from most recent clip: {most_recent_clip_name}")
            else:
                profiles = person_processor.get_person_profiles(clip_path)

            logger.info("CLIP NAME: " + clip_name)
            logger.info("PROFILES: " + str(profiles))
            
            # Get GAIT results for this camera
            gait_results = person_processor.get_gait_results(camera_feed["id"])
            logger.info(f"GAIT results for {camera_feed['id']}: {gait_results}")
            
            for person_id, profile in profiles.items():
                if "analysis" not in profile or "error" in profile:
                    continue
                
                analysis = profile["analysis"]
                
                # Create person object for display
                person = {
                    "id": person_id,
                    "confidence": "87",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "analysis": analysis,
                    "image_url": url_for('get_person_image', clip_name=clip_name, image_type='full', person_id=person_id)
                }
                
                detected_persons.append(person)
    
    return render_template('index.html', 
                           detected_persons=detected_persons,
                           gait_results=gait_results,
                           camera_feeds=camera_feeds,
                           system_status=SYSTEM_STATUS)

@app.route('/tracked-people')
def tracked_people():
    # Get all camera feeds
    camera_feeds = []
    for stream_id, stream_info in stream_manager.streams.items():
        camera_feed = {
            "id": stream_id,
            "name": stream_info.name,
            "status": stream_manager.get_stream_status(stream_id),
            "latest_clip": stream_manager.get_latest_clip(stream_id)
        }
        camera_feeds.append(camera_feed)
    
    # Get tracked subjects from all person profiles
    tracked_subjects = []
    for camera_feed in camera_feeds:
        if camera_feed["latest_clip"]:
            profiles = person_processor.get_person_profiles(camera_feed["latest_clip"])
            
            for person_id, profile in profiles.items():
                if "analysis" not in profile or "error" in profile:
                    continue
                
                analysis = profile["analysis"]
                
                # Create tracked subject
                subject = {
                    "id": person_id,
                    "confidence": "87%",
                    "first_seen": datetime.now().strftime("%H:%M:%S"),
                    "location": camera_feed["name"],
                    "path": [camera_feed["name"]],
                    "color": "#FF0000",
                    "features": [],
                    "recognition_status": "Confirmed",
                    "total_appearances": 1
                }
                
                # Add features from analysis
                if "clothing" in analysis:
                    clothing = analysis["clothing"]
                    for item_type, description in clothing.items():
                        if description:
                            subject["features"].append(description)
                
                if "accessories" in analysis:
                    for accessory in analysis["accessories"]:
                        if "description" in accessory and accessory["description"]:
                            subject["features"].append(accessory["description"])
                
                # Add appearance info
                if "appearance" in analysis:
                    appearance = analysis["appearance"]
                    if "estimated_age_range" in appearance:
                        subject["age_range"] = appearance["estimated_age_range"]
                    if "gender" in appearance:
                        subject["gender"] = appearance["gender"]
                    if "build" in appearance:
                        subject["build"] = appearance["build"]
                
                # Add signature (most distinctive feature)
                if subject["features"]:
                    subject["signature"] = subject["features"][0]
                
                tracked_subjects.append(subject)
    
    return render_template('tracked_people.html',
                           tracked_subjects=tracked_subjects,
                           system_status=SYSTEM_STATUS)

@app.route('/gait-results')
def gait_results():
    """Display GAIT analysis results"""
    # Get camera feeds with stream status and latest clip info
    camera_feeds = []
    for stream_id, stream_info in stream_manager.streams.items():
        camera_feed = {
            "id": stream_id,
            "name": stream_info.name,
            "status": stream_manager.get_stream_status(stream_id),
            "latest_clip": stream_manager.get_latest_clip(stream_id),
            "gait_results": person_processor.get_gait_results(stream_id)
        }
        camera_feeds.append(camera_feed)
    
    return render_template('gait_results.html',
                           camera_feeds=camera_feeds,
                           system_status=SYSTEM_STATUS)

@app.route('/person/<subject_id>')
def person_details(subject_id):
    """Display detailed information about a person"""
    # Get all camera feeds
    camera_feeds = []
    for stream_id, stream_info in stream_manager.streams.items():
        camera_feed = {
            "id": stream_id,
            "name": stream_info.name,
            "latest_clip": stream_manager.get_latest_clip(stream_id)
        }
        camera_feeds.append(camera_feed)
    
    # Find subject in person profiles
    subject = None
    clip_name = None
    
    for camera_feed in camera_feeds:
        if camera_feed["latest_clip"]:
            profiles = person_processor.get_person_profiles(camera_feed["latest_clip"])
            
            if subject_id in profiles:
                profile = profiles[subject_id]
                
                if "analysis" not in profile or "error" in profile:
                    continue
                
                analysis = profile["analysis"]
                clip_name = os.path.splitext(os.path.basename(camera_feed["latest_clip"]))[0]
                
                # Create tracked subject
                subject = {
                    "id": subject_id,
                    "confidence": "87%",
                    "first_seen": datetime.now().strftime("%H:%M:%S"),
                    "location": camera_feed["name"],
                    "path": [camera_feed["name"]],
                    "color": "#FF0000",
                    "features": [],
                    "recognition_status": "Confirmed",
                    "total_appearances": 1,
                    "full_analysis": analysis  # Include full analysis
                }
                
                # Add features from analysis
                if "clothing" in analysis:
                    clothing = analysis["clothing"]
                    for item_type, description in clothing.items():
                        if description:
                            subject["features"].append(description)
                
                if "accessories" in analysis:
                    for accessory in analysis["accessories"]:
                        if "description" in accessory and accessory["description"]:
                            subject["features"].append(accessory["description"])
                
                # Add appearance info
                if "appearance" in analysis:
                    appearance = analysis["appearance"]
                    if "estimated_age_range" in appearance:
                        subject["age_range"] = appearance["estimated_age_range"]
                    if "gender" in appearance:
                        subject["gender"] = appearance["gender"]
                    if "build" in appearance:
                        subject["build"] = appearance["build"]
                
                # Add signature (most distinctive feature)
                if subject["features"]:
                    subject["signature"] = subject["features"][0]
                
                # Add GAIT results
                gait_results = person_processor.get_gait_results(camera_feed["id"])
                if gait_results:
                    subject["gait_results"] = gait_results
                
                break
    
    if not subject:
        return jsonify({"error": "Subject not found"}), 404
    
    # Generate image URLs
    face_image_url = None
    full_image_url = None
    walking_video_url = None
    
    if clip_name:
        face_image_url = url_for('get_person_image', clip_name=clip_name, image_type='face', person_id=subject_id)
        full_image_url = url_for('get_person_image', clip_name=clip_name, image_type='full', person_id=subject_id)
        
        # Check if walking video exists
        profiles_dir = os.path.join("profiles", clip_name)
        videos_dir = os.path.join(profiles_dir, "videos")
        walking_video_path = os.path.join(videos_dir, f"person_{subject_id}_walking.mp4")
        
        if os.path.exists(walking_video_path):
            walking_video_url = url_for('get_walking_video', clip_name=clip_name, person_id=subject_id)
    
    return render_template('person_details.html',
                           subject=subject,
                           face_image_url=face_image_url,
                           full_image_url=full_image_url,
                           walking_video_url=walking_video_url,
                           system_status=SYSTEM_STATUS)

@app.route('/api/subjects')
def get_subjects():
    # Get all camera feeds
    camera_feeds = []
    for stream_id, stream_info in stream_manager.streams.items():
        camera_feed = {
            "id": stream_id,
            "name": stream_info.name,
            "latest_clip": stream_manager.get_latest_clip(stream_id)
        }
        camera_feeds.append(camera_feed)
    
    # Get tracked subjects from all person profiles
    tracked_subjects = []
    for camera_feed in camera_feeds:
        if camera_feed["latest_clip"]:
            profiles = person_processor.get_person_profiles(camera_feed["latest_clip"])
            
            for person_id, profile in profiles.items():
                if "analysis" not in profile or "error" in profile:
                    continue
                
                analysis = profile["analysis"]
                
                # Create tracked subject
                subject = {
                    "id": person_id,
                    "confidence": "87%",
                    "first_seen": datetime.now().strftime("%H:%M:%S"),
                    "location": camera_feed["name"],
                    "path": [camera_feed["name"]],
                    "color": "#FF0000",
                    "features": [],
                    "recognition_status": "Confirmed",
                    "total_appearances": 1
                }
                
                # Add features from analysis
                if "clothing" in analysis:
                    clothing = analysis["clothing"]
                    for item_type, description in clothing.items():
                        if description:
                            subject["features"].append(description)
                
                if "accessories" in analysis:
                    for accessory in analysis["accessories"]:
                        if "description" in accessory and accessory["description"]:
                            subject["features"].append(accessory["description"])
                
                # Add appearance info
                if "appearance" in analysis:
                    appearance = analysis["appearance"]
                    if "estimated_age_range" in appearance:
                        subject["age_range"] = appearance["estimated_age_range"]
                    if "gender" in appearance:
                        subject["gender"] = appearance["gender"]
                    if "build" in appearance:
                        subject["build"] = appearance["build"]
                
                # Add signature (most distinctive feature)
                if subject["features"]:
                    subject["signature"] = subject["features"][0]
                
                tracked_subjects.append(subject)
    
    return jsonify(tracked_subjects)

@app.route('/api/cameras')
def get_cameras():
    # Get camera feeds with stream status and latest clip info
    camera_feeds = []
    for stream_id, stream_info in stream_manager.streams.items():
        camera_feed = {
            "id": stream_id,
            "name": stream_info.name,
            "status": stream_manager.get_stream_status(stream_id),
            "detected_subjects": [],  # Will be populated from person profiles
            "video_id": stream_info.video_id,
            "latest_clip": stream_manager.get_latest_clip(stream_id),
            "has_clip": False
        }
        
        # Check if clip exists
        if camera_feed["latest_clip"] and os.path.exists(camera_feed["latest_clip"]):
            camera_feed["has_clip"] = True
            
            # Get person profiles for this clip
            profiles = person_processor.get_person_profiles(camera_feed["latest_clip"])
            for person_id in profiles:
                camera_feed["detected_subjects"].append(person_id)
        
        camera_feeds.append(camera_feed)
    
    return jsonify(camera_feeds)

@app.route('/api/system')
def get_system_status():
    return jsonify(SYSTEM_STATUS)

@app.route('/api/subject/<subject_id>')
def get_subject(subject_id):
    # Get all camera feeds
    camera_feeds = []
    for stream_id, stream_info in stream_manager.streams.items():
        camera_feed = {
            "id": stream_id,
            "name": stream_info.name,
            "latest_clip": stream_manager.get_latest_clip(stream_id)
        }
        camera_feeds.append(camera_feed)
    
    # Find subject in person profiles
    for camera_feed in camera_feeds:
        if camera_feed["latest_clip"]:
            profiles = person_processor.get_person_profiles(camera_feed["latest_clip"])
            
            if subject_id in profiles:
                profile = profiles[subject_id]
                
                if "analysis" not in profile or "error" in profile:
                    continue
                
                analysis = profile["analysis"]
                
                # Create tracked subject
                subject = {
                    "id": subject_id,
                    "confidence": "87%",
                    "first_seen": datetime.now().strftime("%H:%M:%S"),
                    "location": camera_feed["name"],
                    "path": [camera_feed["name"]],
                    "color": "#FF0000",
                    "features": [],
                    "recognition_status": "Confirmed",
                    "total_appearances": 1,
                    "full_analysis": analysis  # Include full analysis
                }
                
                # Add features from analysis
                if "clothing" in analysis:
                    clothing = analysis["clothing"]
                    for item_type, description in clothing.items():
                        if description:
                            subject["features"].append(description)
                
                if "accessories" in analysis:
                    for accessory in analysis["accessories"]:
                        if "description" in accessory and accessory["description"]:
                            subject["features"].append(accessory["description"])
                
                # Add appearance info
                if "appearance" in analysis:
                    appearance = analysis["appearance"]
                    if "estimated_age_range" in appearance:
                        subject["age_range"] = appearance["estimated_age_range"]
                    if "gender" in appearance:
                        subject["gender"] = appearance["gender"]
                    if "build" in appearance:
                        subject["build"] = appearance["build"]
                
                # Add signature (most distinctive feature)
                if subject["features"]:
                    subject["signature"] = subject["features"][0]
                
                # Add GAIT results
                gait_results = person_processor.get_gait_results(camera_feed["id"])
                if gait_results:
                    subject["gait_results"] = gait_results
                
                return jsonify(subject)
    
    return jsonify({"error": "Subject not found"}), 404

@app.route('/api/gait/<stream_id>')
def get_gait_results(stream_id):
    """Get GAIT detection results for a stream"""
    gait_results = person_processor.get_gait_results(stream_id)
    return jsonify(gait_results)

@app.route('/api/profile/<clip_name>/<person_id>')
def get_person_profile(clip_name, person_id):
    """Get profile for a specific person in a clip"""
    # Find the profiles directory for this clip
    profiles_dir = os.path.join("profiles", clip_name)
    
    # Check if directory exists
    if not os.path.exists(profiles_dir):
        return jsonify({"error": "Profile not found"}), 404
    
    # Check if Claude analysis exists
    claude_analysis_path = os.path.join(profiles_dir, "claude_analysis.json")
    if not os.path.exists(claude_analysis_path):
        return jsonify({"error": "Analysis not found"}), 404
    
    # Load Claude analysis
    with open(claude_analysis_path, 'r') as f:
        claude_analysis = json.load(f)
    
    # Check if person exists
    if person_id not in claude_analysis:
        return jsonify({"error": "Person not found"}), 404
    
    return jsonify(claude_analysis[person_id])

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

# Route to get a video stream for a camera (for continuous playback)
@app.route('/api/video/stream/<camera_id>')
def stream_video_feed(camera_id):
    def generate():
        while True:
            # Get the latest clip for this camera
            latest_clip = stream_manager.get_latest_clip(camera_id)
            
            if latest_clip and os.path.exists(latest_clip):
                # Read the video file in chunks
                with open(latest_clip, 'rb') as f:
                    while True:
                        chunk = f.read(1024)
                        if not chunk:
                            break
                        yield chunk
                
                # Wait a short time before checking for a new clip
                time.sleep(0.5)
            else:
                # If no clip is available, wait a bit and try again
                time.sleep(1)
    
    return Response(generate(), mimetype='video/mp4')

# Route to get a walking video
@app.route('/api/walking-video/<clip_name>/<person_id>')
def get_walking_video(clip_name, person_id):
    """Get walking video for a specific person in a clip"""
    # Find the profiles directory for this clip
    profiles_dir = os.path.join("profiles", clip_name)
    videos_dir = os.path.join(profiles_dir, "videos")
    
    # Check if directory exists
    if not os.path.exists(videos_dir):
        return jsonify({"error": "Videos directory not found"}), 404
    
    # Determine video path
    video_path = os.path.join(videos_dir, f"person_{person_id}_walking.mp4")
    
    # Check if video exists
    if not os.path.exists(video_path):
        return jsonify({"error": "Walking video not found"}), 404
    
    # Return the video file
    return send_file(video_path, mimetype='video/mp4')

# Route to get a person image
@app.route('/api/image/<clip_name>/<image_type>/<person_id>')
def get_person_image(clip_name, image_type, person_id):
    """Get image for a specific person in a clip"""
    # Find the profiles directory for this clip
    profiles_dir = os.path.join("profiles", clip_name)
    
    # Debug logging
    logger.info(f"Image request: {clip_name}, {image_type}, {person_id}")
    logger.info(f"Profiles dir exists: {os.path.exists(profiles_dir)}")
    
    # Check if directory exists
    if not os.path.exists(profiles_dir):
        logger.error(f"Profile directory not found: {profiles_dir}")
        return jsonify({"error": "Profile not found"}), 404
    
    # Determine image path based on type
    if image_type == "face":
        image_dir = os.path.join(profiles_dir, "faces")
        image_path = os.path.join(image_dir, f"person_{person_id}_face.jpg")
    elif image_type == "full":
        image_dir = os.path.join(profiles_dir, "full_bodies")
        image_path = os.path.join(image_dir, f"person_{person_id}_full.jpg")
    else:
        return jsonify({"error": "Invalid image type"}), 400
    
    # More debug logging
    logger.info(f"Image dir exists: {os.path.exists(image_dir)}")
    logger.info(f"Image path: {image_path}")
    logger.info(f"Image exists: {os.path.exists(image_path)}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return jsonify({"error": "Image not found"}), 404
    
    # Return the image file
    return send_file(image_path, mimetype='image/jpeg')

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

# Add a before request handler to restart services if needed
@app.before_request
def before_request():
    # Skip if request is for static files
    if request.path.startswith('/static/'):
        return None
    
    # Ensure services are running before processing requests
    if not stream_manager.is_running:
        try:
            stream_manager.start()
            logger.info("Restarted stream manager before handling request")
        except Exception as e:
            logger.error(f"Failed to restart stream manager: {e}")

# Remove the teardown_appcontext handler that was stopping services
# This was causing premature shutdown after each request
@app.teardown_appcontext
def cleanup(exception=None):
    # Don't actually stop services on every request!
    # Only log errors if they occur
    if exception:
        logger.error(f"Error during request: {exception}")

# Register proper shutdown handler for when the app actually exits
@atexit.register
def shutdown():
    global services_running
    services_running = False
    logger.info("Application shutting down - stopping services")
    
    # Stop the services
    try:
        stream_manager.stop()
    except Exception as e:
        logger.error(f"Error stopping stream manager: {e}")
    
    try:
        person_processor.stop()
    except Exception as e:
        logger.error(f"Error stopping person processor: {e}")
    
    logger.info("All services stopped")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)