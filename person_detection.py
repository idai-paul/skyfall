import cv2
import torch
import os
import numpy as np
from collections import defaultdict

# Create profiles directory if it doesn't exist
profiles_dir = "profiles"
if not os.path.exists(profiles_dir):
    os.makedirs(profiles_dir)

# Load pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load face detector (using OpenCV's built-in face detector)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open video
video_path = 'youtube_clips/clip_20250426_135130.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Simple tracker using IoU for matching detections across frames
class SimpleTracker:
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = {}  # Tracked objects: id -> bbox
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.person_frames = defaultdict(list)
        self.full_body_saved = set()
        self.face_saved = set()
        self.video_writers = {}
        
    def _calculate_iou(self, boxA, boxB):
        # Determine the coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)
        
        # Compute the area of both bounding boxes
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        # Compute the IoU
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    def update(self, detections, frame):
        # If no detections, mark all objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.objects.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._remove_object(object_id)
            return self.objects
        
        # If we're tracking nothing, register all detections
        if len(self.objects) == 0:
            for detection in detections:
                self._register(detection, frame)
            return self.objects
        
        # Match detections to existing objects
        object_ids = list(self.objects.keys())
        object_bboxes = list(self.objects.values())
        
        used_objects = set()
        used_detections = set()
        
        # For each detection, find the best matching object
        for i, detection in enumerate(detections):
            max_iou = 0.3  # IoU threshold
            best_match = None
            
            for j, object_bbox in enumerate(object_bboxes):
                object_id = object_ids[j]
                
                iou = self._calculate_iou(detection, object_bbox)
                if iou > max_iou:
                    max_iou = iou
                    best_match = j
            
            if best_match is not None:
                object_id = object_ids[best_match]
                self.objects[object_id] = detection
                self.disappeared[object_id] = 0
                used_objects.add(best_match)
                used_detections.add(i)
                
                # Store frame for this person
                person_img = self._extract_person(frame, detection)
                self.person_frames[object_id].append(person_img)
                
                # Save face if available
                if object_id not in self.face_saved:
                    face_img = self._extract_face(person_img)
                    if face_img is not None:
                        face_path = os.path.join(profiles_dir, f"person_{object_id}_face.jpg")
                        cv2.imwrite(face_path, face_img)
                        self.face_saved.add(object_id)
                
                # Save full body if not already saved
                if object_id not in self.full_body_saved:
                    full_body_path = os.path.join(profiles_dir, f"person_{object_id}_full.jpg")
                    cv2.imwrite(full_body_path, person_img)
                    self.full_body_saved.add(object_id)
                
                # Create video writer if it doesn't exist
                if object_id not in self.video_writers:
                    video_path = os.path.join(profiles_dir, f"person_{object_id}_walking.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    width = person_img.shape[1]
                    height = person_img.shape[0]
                    self.video_writers[object_id] = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                
                # Write frame to video
                self.video_writers[object_id].write(person_img)
        
        # Register new detections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                self._register(detection, frame)
        
        # Update disappeared counts for missing objects
        for j, object_id in enumerate(object_ids):
            if j not in used_objects:
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._remove_object(object_id)
        
        return self.objects
    
    def _register(self, bbox, frame):
        # Register new object
        object_id = self.next_id
        self.next_id += 1
        self.objects[object_id] = bbox
        self.disappeared[object_id] = 0
        
        # Extract and store person image
        person_img = self._extract_person(frame, bbox)
        self.person_frames[object_id].append(person_img)
        
        # Save full body image
        full_body_path = os.path.join(profiles_dir, f"person_{object_id}_full.jpg")
        cv2.imwrite(full_body_path, person_img)
        self.full_body_saved.add(object_id)
        
        # Try to save face
        face_img = self._extract_face(person_img)
        if face_img is not None:
            face_path = os.path.join(profiles_dir, f"person_{object_id}_face.jpg")
            cv2.imwrite(face_path, face_img)
            self.face_saved.add(object_id)
        
        # Create video writer
        video_path = os.path.join(profiles_dir, f"person_{object_id}_walking.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = person_img.shape[1]
        height = person_img.shape[0]
        self.video_writers[object_id] = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        self.video_writers[object_id].write(person_img)
    
    def _remove_object(self, object_id):
        # Close video writer if it exists
        if object_id in self.video_writers:
            self.video_writers[object_id].release()
            del self.video_writers[object_id]
        
        # Remove object from tracking
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def _extract_person(self, frame, bbox):
        # Extract person from frame using bounding box
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # Ensure within frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        return frame[y1:y2, x1:x2].copy()
    
    def _extract_face(self, person_img):
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            # Get largest face
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            x, y, w, h = largest_face
            
            # Extract face
            face_img = person_img[y:y+h, x:x+w].copy()
            return face_img
        return None
    
    def cleanup(self):
        # Release all video writers
        for object_id in self.video_writers:
            self.video_writers[object_id].release()

# Initialize tracker
tracker = SimpleTracker()

frame_count = 0

while cap.isOpened():
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Run detection
    results = model(frame)
    
    # Filter for only people (class 0 in COCO dataset)
    people_results = results.xyxy[0][results.xyxy[0][:, 5] == 0].cpu().numpy()
    
    # Get bounding boxes
    boxes = []
    for box in people_results:
        x1, y1, x2, y2, conf, cls = box
        if conf > 0.5:  # Confidence threshold
            boxes.append([x1, y1, x2, y2])
    
    # Update tracker
    tracked_objects = tracker.update(boxes, frame)
    
    # Draw bounding boxes and IDs
    for object_id, bbox in tracked_objects.items():
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {object_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display result
    cv2.imshow('People Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
tracker.cleanup()
cap.release()
cv2.destroyAllWindows()