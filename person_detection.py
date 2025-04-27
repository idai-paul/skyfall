import cv2
import torch
import os
import numpy as np
from collections import defaultdict

class PersonProfiler:
    def __init__(self, video_path, min_confidence=0.5, min_size=50, max_disappeared=30, 
                 min_tracking_seconds=3, max_tracking_seconds=5, box_expansion_ratio=0.15):
        """
        Initialize the person profiling system
        
        Args:
            video_path: Path to the video file
            min_confidence: Minimum confidence threshold for detections
            min_size: Minimum size in pixels for a detection to be tracked
            max_disappeared: Maximum number of frames an object can disappear before being removed
            min_tracking_seconds: Minimum seconds to track a person before creating video
            max_tracking_seconds: Maximum seconds to keep tracking data
            box_expansion_ratio: Ratio to expand bounding boxes (0.15 = 15% expansion)
        """
        # Create clip-specific directory for profiles
        self.video_path = video_path
        self.clip_name = os.path.splitext(os.path.basename(video_path))[0]
        self.profiles_dir = os.path.join("profiles", self.clip_name)
        os.makedirs(self.profiles_dir, exist_ok=True)
        
        # Create subdirectories for videos, faces, and full-body images
        self.videos_dir = os.path.join(self.profiles_dir, "videos")
        self.faces_dir = os.path.join(self.profiles_dir, "faces")
        self.full_bodies_dir = os.path.join(self.profiles_dir, "full_bodies")
        
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.faces_dir, exist_ok=True)
        os.makedirs(self.full_bodies_dir, exist_ok=True)
        
        # Initialize models
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Video properties
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Tracking parameters
        self.min_confidence = min_confidence
        self.min_frames = int(self.fps * min_tracking_seconds)
        self.max_frames = int(self.fps * max_tracking_seconds)
        self.box_expansion_ratio = box_expansion_ratio
        
        # Initialize tracker
        self.tracker = self.SimpleTracker(
            max_disappeared=max_disappeared,
            min_frames=self.min_frames,
            max_frames=self.max_frames,
            min_size=min_size,
            profiles_dir=self.profiles_dir,
            videos_dir=self.videos_dir,
            faces_dir=self.faces_dir,
            full_bodies_dir=self.full_bodies_dir,
            fps=self.fps,
            box_expansion_ratio=box_expansion_ratio
        )
        
    def process_video(self, display=True):
        """
        Process the entire video file and track people
        
        Args:
            display: Whether to display the processing in a window
        """
        frame_count = 0
        
        while self.cap.isOpened():
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Processing frame {frame_count}")
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display result
            if display:
                cv2.imshow('People Detection', processed_frame)
                if cv2.waitKey(1) == ord('q'):
                    break
        
        # Ensure all videos are finalized
        self.tracker.cleanup()
        self.cap.release()
        cv2.destroyAllWindows()
        
    def process_frame(self, frame):
        """
        Process a single frame to detect and track people
        
        Args:
            frame: The video frame to process
            
        Returns:
            Processed frame with bounding boxes and tracking info
        """
        # Run detection
        results = self.model(frame)
        
        # Filter for only people (class 0 in COCO dataset)
        people_results = results.xyxy[0][results.xyxy[0][:, 5] == 0].cpu().numpy()
        
        # Get bounding boxes
        boxes = []
        for box in people_results:
            x1, y1, x2, y2, conf, cls = box
            if conf > self.min_confidence:
                boxes.append([x1, y1, x2, y2])
        
        # Update tracker
        tracked_objects = self.tracker.update(boxes, frame.copy())
        
        # Draw bounding boxes and IDs
        for object_id, bbox in tracked_objects.items():
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add ID and walking status
            is_walking = self.tracker.is_walking(object_id)
            status = "WALKING" if is_walking else "STATIC"
            cv2.putText(frame, f"ID: {object_id} ({status})", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    class SimpleTracker:
        def __init__(self, max_disappeared=30, min_frames=90, max_frames=150, min_size=50, 
                     profiles_dir="profiles", videos_dir=None, faces_dir=None, full_bodies_dir=None,
                     fps=30, box_expansion_ratio=0.15):
            """
            Initialize the tracker
            
            Args:
                max_disappeared: Maximum frames an object can disappear before tracking stops
                min_frames: Minimum frames required to create a walking video
                max_frames: Maximum frames to keep in buffer
                min_size: Minimum size in pixels for a good detection
                profiles_dir: Directory to save profiles
                fps: Frames per second of the video
                box_expansion_ratio: Ratio to expand bounding boxes (0.15 = 15% expansion)
            """
            self.next_id = 0
            self.objects = {}  # Tracked objects: id -> bbox
            self.disappeared = defaultdict(int)
            self.max_disappeared = max_disappeared
            self.min_frames = min_frames
            self.max_frames = max_frames
            self.min_size = min_size
            self.full_body_saved = set()
            self.face_saved = set()
            self.profiles_dir = profiles_dir
            self.videos_dir = videos_dir or os.path.join(profiles_dir, "videos")
            self.faces_dir = faces_dir or os.path.join(profiles_dir, "faces")
            self.full_bodies_dir = full_bodies_dir or os.path.join(profiles_dir, "full_bodies")
            self.fps = fps
            self.box_expansion_ratio = box_expansion_ratio
            
            # Buffer of original frames for each person
            self.frame_buffer = {}  # id -> list of (frame, bbox)
            # Track movement to identify walking
            self.position_history = defaultdict(list)
            
        def calculate_iou(self, boxA, boxB):
            """
            Calculate the Intersection over Union between two bounding boxes
            
            Args:
                boxA: First bounding box [x1, y1, x2, y2]
                boxB: Second bounding box [x1, y1, x2, y2]
                
            Returns:
                IoU value between 0 and 1
            """
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
        
        def update(self, detections, original_frame):
            """
            Update tracker with new detections
            
            Args:
                detections: List of detected bounding boxes [x1, y1, x2, y2]
                original_frame: Original frame for storing
                
            Returns:
                Dictionary of tracked objects {id -> bbox}
            """
            # If no detections, mark all objects as disappeared
            if len(detections) == 0:
                for object_id in list(self.objects.keys()):
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.remove_object(object_id)
                return self.objects
            
            # If we're tracking nothing, register all detections
            if len(self.objects) == 0:
                for detection in detections:
                    self.register(detection, original_frame)
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
                    
                    iou = self.calculate_iou(detection, object_bbox)
                    if iou > max_iou:
                        max_iou = iou
                        best_match = j
                
                if best_match is not None:
                    object_id = object_ids[best_match]
                    self.objects[object_id] = detection
                    self.disappeared[object_id] = 0
                    used_objects.add(best_match)
                    used_detections.add(i)
                    
                    # Get detection size to filter small objects
                    x1, y1, x2, y2 = detection
                    width = x2 - x1
                    height = y2 - y1
                    size = max(width, height)
                    
                    # Track center position for walking detection
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    self.position_history[object_id].append((center_x, center_y))
                    
                    # Only keep the last 30 positions
                    if len(self.position_history[object_id]) > 30:
                        self.position_history[object_id].pop(0)
                    
                    # Only track sufficiently large detections
                    if size >= self.min_size:
                        # Store original frame and bbox
                        if object_id in self.frame_buffer:
                            # Add to the buffer, but maintain target size
                            self.frame_buffer[object_id].append((original_frame.copy(), detection))
                            
                            # If we have more than max frames, remove oldest
                            while len(self.frame_buffer[object_id]) > self.max_frames:
                                self.frame_buffer[object_id].pop(0)
                        
                        # Extract person image for profile
                        person_img = self.extract_person(original_frame, detection)
                        if person_img.size > 0:
                            # Save face if available and not already saved
                            if object_id not in self.face_saved:
                                face_img = self.extract_face(person_img)
                                if face_img is not None and face_img.size > 0:
                                    face_path = os.path.join(self.faces_dir, f"person_{object_id}_face.jpg")
                                    cv2.imwrite(face_path, face_img)
                                    self.face_saved.add(object_id)
                            
                            # Save full body if not already saved
                            if object_id not in self.full_body_saved:
                                full_body_path = os.path.join(self.full_bodies_dir, f"person_{object_id}_full.jpg")
                                cv2.imwrite(full_body_path, person_img)
                                self.full_body_saved.add(object_id)
            
            # Register new detections
            for i, detection in enumerate(detections):
                if i not in used_detections:
                    self.register(detection, original_frame)
            
            # Update disappeared counts for missing objects
            for j, object_id in enumerate(object_ids):
                if j not in used_objects:
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.remove_object(object_id)
            
            return self.objects
        
        def register(self, bbox, original_frame):
            """
            Register a new object for tracking
            
            Args:
                bbox: Bounding box [x1, y1, x2, y2]
                original_frame: Original frame for storing
            """
            # Check if the detection is large enough
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            size = max(width, height)
            
            if size < self.min_size:
                return  # Skip small detections
                
            # Register new object
            object_id = self.next_id
            self.next_id += 1
            self.objects[object_id] = bbox
            self.disappeared[object_id] = 0
            
            # Initialize position history
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            self.position_history[object_id].append((center_x, center_y))
            
            # Initialize frame buffer for this person
            self.frame_buffer[object_id] = [(original_frame.copy(), bbox)]
            
            # Extract and save profile images
            person_img = self.extract_person(original_frame, bbox)
            if person_img.size > 0:
                # Save full body image
                full_body_path = os.path.join(self.full_bodies_dir, f"person_{object_id}_full.jpg")
                cv2.imwrite(full_body_path, person_img)
                self.full_body_saved.add(object_id)
                
                # Try to save face
                face_img = self.extract_face(person_img)
                if face_img is not None and face_img.size > 0:
                    face_path = os.path.join(self.faces_dir, f"person_{object_id}_face.jpg")
                    cv2.imwrite(face_path, face_img)
                    self.face_saved.add(object_id)
        
        def remove_object(self, object_id):
            """
            Remove an object from tracking and create video if criteria met
            
            Args:
                object_id: ID of the object to remove
            """
            # Check if the person has been walking
            is_walking = self.is_walking(object_id)
            
            # Create video only if:
            # 1. We have enough frames (at least 3 seconds)
            # 2. The person was actually walking
            # 3. The person was large enough in the frame
            if (object_id in self.frame_buffer and 
                len(self.frame_buffer[object_id]) >= self.min_frames and
                is_walking):
                self.create_walking_video(object_id)
            
            # Remove object from tracking
            if object_id in self.objects:
                del self.objects[object_id]
            if object_id in self.disappeared:
                del self.disappeared[object_id]
            if object_id in self.frame_buffer:
                del self.frame_buffer[object_id]
            if object_id in self.position_history:
                del self.position_history[object_id]
        
        def is_walking(self, object_id):
            """
            Determine if the person is walking based on their position history
            
            Args:
                object_id: ID of the object to check
                
            Returns:
                Boolean indicating if the person is walking
            """
            if object_id not in self.position_history:
                return False
                
            positions = self.position_history[object_id]
            if len(positions) < 10:  # Need at least 10 positions to determine walking
                return False
                
            # Calculate total distance moved
            total_distance = 0
            for i in range(1, len(positions)):
                x1, y1 = positions[i-1]
                x2, y2 = positions[i]
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                total_distance += distance
                
            # Calculate average speed (pixels per frame)
            avg_speed = total_distance / (len(positions) - 1)
            
            # Check if the movement is consistent and significant
            return avg_speed > 2.0  # Threshold for walking speed
        
        def expand_bbox(self, bbox, frame_width, frame_height):
            """
            Expand the bounding box by the expansion ratio
            
            Args:
                bbox: Original bounding box [x1, y1, x2, y2]
                frame_width: Width of the frame
                frame_height: Height of the frame
                
            Returns:
                Expanded bounding box [x1, y1, x2, y2]
            """
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # Calculate expansion amount
            width_expansion = width * self.box_expansion_ratio
            height_expansion = height * self.box_expansion_ratio
            
            # Expand the box
            expanded_x1 = max(0, x1 - width_expansion)
            expanded_y1 = max(0, y1 - height_expansion)
            expanded_x2 = min(frame_width, x2 + width_expansion)
            expanded_y2 = min(frame_height, y2 + height_expansion)
            
            return [expanded_x1, expanded_y1, expanded_x2, expanded_y2]
        
        def extract_person(self, frame, bbox):
            """
            Extract person image from frame using bounding box with expansion
            
            Args:
                frame: Video frame
                bbox: Bounding box [x1, y1, x2, y2]
                
            Returns:
                Cropped image of the person with expanded bounding box
            """
            # Expand the bounding box
            expanded_bbox = self.expand_bbox(bbox, frame.shape[1], frame.shape[0])
            
            # Extract person from frame using expanded bounding box
            x1, y1, x2, y2 = [int(coord) for coord in expanded_bbox]
            
            # Ensure within frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # Check if the bounding box is valid (non-zero area)
            if x2 <= x1 or y2 <= y1:
                return np.array([])
                
            return frame[y1:y2, x1:x2].copy()
        
        def extract_face(self, person_img):
            """
            Extract face image from person image
            
            Args:
                person_img: Image of the person
                
            Returns:
                Cropped image of the face or None if no face detected
            """
            # Check if the image is valid
            if person_img.size == 0 or person_img.shape[0] == 0 or person_img.shape[1] == 0:
                return None
                
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                # Get largest face
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                x, y, w, h = largest_face
                
                # Ensure the face region is within bounds
                if x >= 0 and y >= 0 and x+w <= person_img.shape[1] and y+h <= person_img.shape[0]:
                    # Extract face
                    face_img = person_img[y:y+h, x:x+w].copy()
                    return face_img
            return None
        
        def create_walking_video(self, object_id):
            """
            Create a video clip of the person walking
            
            Args:
                object_id: ID of the object
            """
            buffer = self.frame_buffer[object_id]
            
            print(f"Creating {len(buffer)}-frame video for person {object_id} (approx {len(buffer)/self.fps:.1f} seconds)")
            
            # Get the first frame and box to determine dimensions
            first_frame, first_box = buffer[0]
            
            # Expand the first bounding box
            expanded_box = self.expand_bbox(first_box, first_frame.shape[1], first_frame.shape[0])
            x1, y1, x2, y2 = map(int, expanded_box)
            
            # For consistent video size, use the expanded bounding box dimensions
            width = x2 - x1
            height = y2 - y1
            
            # Add padding to avoid size issues when the person moves
            padding = int(max(width, height) * 0.1)  # 10% padding
            width += 2 * padding
            height += 2 * padding
            
            # Create video file
            video_path = os.path.join(self.videos_dir, f"person_{object_id}_walking.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Try 'avc1' or 'XVID' if mp4v doesn't work
            out = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
            
            # Write frames
            for frame, bbox in buffer:
                # Expand the bounding box
                expanded_bbox = self.expand_bbox(bbox, frame.shape[1], frame.shape[0])
                x1, y1, x2, y2 = map(int, expanded_bbox)
                
                # Calculate center of the person
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Calculate new bounding box with padding
                new_x1 = max(0, center_x - width // 2)
                new_y1 = max(0, center_y - height // 2)
                new_x2 = min(frame.shape[1], new_x1 + width)
                new_y2 = min(frame.shape[0], new_y1 + height)
                
                # Adjust if near frame edges
                if new_x2 - new_x1 < width:
                    if new_x1 == 0:
                        new_x2 = min(frame.shape[1], new_x1 + width)
                    else:
                        new_x1 = max(0, new_x2 - width)
                
                if new_y2 - new_y1 < height:
                    if new_y1 == 0:
                        new_y2 = min(frame.shape[0], new_y1 + height)
                    else:
                        new_y1 = max(0, new_y2 - height)
                
                # Extract region
                person_img = frame[new_y1:new_y2, new_x1:new_x2].copy()
                
                # Resize if necessary
                if person_img.shape[1] != width or person_img.shape[0] != height:
                    person_img = cv2.resize(person_img, (width, height))
                
                # Write to video
                out.write(person_img)
            
            # Release writer
            out.release()
            print(f"Video saved for person {object_id} with {len(buffer)} frames (~{len(buffer)/self.fps:.1f} seconds)")
        
        def cleanup(self):
            """
            Finalize tracking and create videos for all tracked people that meet criteria
            """
            # Create videos for all tracked people that meet our criteria
            for object_id in list(self.frame_buffer.keys()):
                if (len(self.frame_buffer[object_id]) >= self.min_frames and 
                    self.is_walking(object_id)):
                    self.create_walking_video(object_id)


# Main execution function
def main(video_path="youtube_clips/clip_20250426_152122.mp4", display=True, box_expansion_ratio=0.15):
    """
    Main execution function
    
    Args:
        video_path: Path to the video file
        display: Whether to display processing in a window
        box_expansion_ratio: Ratio to expand bounding boxes (0.15 = 15% expansion)
    """
    profiler = PersonProfiler(
        video_path=video_path,
        min_confidence=0.5,
        min_size=50,
        max_disappeared=30,
        min_tracking_seconds=3,
        max_tracking_seconds=5,
        box_expansion_ratio=box_expansion_ratio
    )
    
    profiler.process_video(display=display)
    
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Process video for person profiling with expanded bounding boxes')
    parser.add_argument('--video', type=str, default="youtube_clips/CAM-12/clip_20250426_213821.mp4", help='Path to video file')
    parser.add_argument('--display', action='store_true', help='Display processing in window')
    parser.add_argument('--expansion', type=float, default=0.15, help='Bounding box expansion ratio (e.g., 0.15 for 15%)')
    
    args = parser.parse_args()
    
    main(video_path=args.video, display=args.display, box_expansion_ratio=args.expansion)
