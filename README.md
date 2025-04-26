# skyfall
National Security Hackathon

## Project Structure

```
skyfall/
├── app.py                  # Main Flask application
├── person_tracker.py       # Person tracking and clustering logic
├── requirements.txt        # Python dependencies
├── static/                 # Static files (CSS, JS, demo videos)
│   ├── css/
│   ├── js/
│   └── demo_videos/        # Sample video feeds for testing
├── templates/              # HTML templates
│   └── index.html          # Main interface template
└── README.md               # Project documentation
```

## Setup Instructions

1. Create a new Python virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create the requirements.txt file with the following contents:
   ```
   flask==2.2.3
   opencv-python==4.7.0.72
   numpy==1.24.2
   scikit-learn==1.2.2
   ```

5. Create the necessary directories:
   ```
   mkdir -p static/demo_videos
   mkdir -p templates
   ```

6. Place your video files in the `static/demo_videos/` directory or use webcams.

7. Run the application:
   ```
   python app.py
   ```

8. Open your browser and navigate to `http://localhost:5000`

## Adding Real Video Sources

To use real video sources instead of the simulated ones:

1. Replace the webcam indexes or video file paths in the `VIDEO_SOURCES` dictionary in `app.py`:
   ```python
   VIDEO_SOURCES = {
       'A1': {'name': 'Entrance A', 'source': 'rtsp://user:pass@192.168.1.100:554/stream1'},
       'A2': {'name': 'Lobby', 'source': 'rtsp://user:pass@192.168.1.101:554/stream1'},
       # Add more cameras as needed
   }
   ```

2. For IP cameras, use RTSP URLs in the format: `rtsp://username:password@ip_address:port/stream_path`

## Implementing a Real Person Detection Model

To replace the simulated person detection with a real model:

1. Install additional dependencies:
   ```
   pip install tensorflow==2.12.0
   ```
   or
   ```
   pip install torch==2.0.0
   ```

2. Modify the `detect_people()` function in `app.py` to use a real model like YOLOv5 or SSD.

3. Example implementation with YOLOv5:
   ```python
   # First install YOLOv5: pip install yolov5
   import yolov5

   # Load the model
   model = yolov5.load('yolov5s.pt')
   model.conf = 0.25  # confidence threshold
   model.classes = [0]  # only detect persons (class 0)

   def detect_people(frame, camera_id):
       """Detect people in a frame using YOLOv5"""
       processed_frame = frame.copy()
       
       # Run detection
       results = model(frame)
       
       # Process detections
       detections = []
       for pred in results.pred[0]:
           if pred[5] == 0:  # Class 0 is person
               x1, y1, x2, y2 = pred[:4].int().cpu().numpy()
               confidence = float(pred[4]) * 100
               
               # Draw the bounding box
               cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
               
               # Create detection object
               detection = {
                   'bbox': [int(x1), int(y1), int(x2), int(y2)],
                   'timestamp': datetime.now().strftime('%H:%M:%S'),
                   'features': extract_features(frame, [int(x1), int(y1), int(x2), int(y2)]),
                   'confidence': int(confidence)
               }
               
               detections.append(detection)
               
               # Label the detection
               cv2.putText(processed_frame, f"Person {confidence:.1f}%", 
                         (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
       
       return processed_frame, detections
   ```

## Implementing a Real Feature Extractor

To replace the simulated feature extraction with a real model:

1. Install additional dependencies:
   ```
   pip install tensorflow==2.12.0
   ```

2. Modify the `extract_features()` function in `app.py`:
   ```python
   import tensorflow as tf
   from tensorflow.keras.applications import ResNet50
   from tensorflow.keras.applications.resnet import preprocess_input
   from tensorflow.keras.models import Model

   # Load the model (once at startup)
   base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
   feature_model = Model(inputs=base_model.input, outputs=base_model.output)

   def extract_features(frame, bbox):
       """Extract features from a person detection using ResNet50"""
       x1, y1, x2, y2 = bbox
       
       # Crop the person from the frame
       person_img = frame[y1:y2, x1:x2]
       
       # If the crop is empty, return random features
       if person_img.size == 0:
           return np.random.rand(2048).tolist()
       
       # Resize to the model's input size
       person_img = cv2.resize(person_img, (224, 224))
       
       # Convert to RGB (OpenCV uses BGR)
       person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
       
       # Preprocess
       person_img = preprocess_input(np.expand_dims(person_img, axis=0))
       
       # Extract features
       features = feature_model.predict(person_img)[0]
       
       return features.tolist()
   ```

## Customizing the UI

The UI can be customized by modifying the HTML and CSS in `templates/index.html`. The main components you might want to customize:

1. Camera grid layout: Adjust the `grid-cols-3` class in the camera grid div to change the number of columns.

2. Map layout: Modify the room positions in the facility map div.

3. Feature colors: Change the colors in the `cluster_colors` array in `person_tracker.py`.

## Performance Optimization

For better performance with multiple cameras:

1. Reduce the frame processing rate in the `process_camera_feed()` function by adjusting the sleep time.

2. Process frames at a lower resolution:
   ```python
   # Resize frame for processing
   frame_small = cv2.resize(frame, (640, 360))
   processed_frame, detections = detect_people(frame_small, camera_id)
   
   # But display the original resolution
   if not frame_queues[camera_id].full():
       display_frame = frame.copy()
       # Scale up detection boxes from small frame to original frame
       # [add scaling code here]
       frame_queues[camera_id].put(display_frame)
   ```

3. Run the person detection on a separate thread for each camera to avoid blocking.

## Troubleshooting

1. **Camera feeds not showing**: Check that the camera sources are accessible and the paths are correct.

2. **High CPU usage**: Reduce the frame rate or resolution of the video processing.

3. **OpenCV errors**: Make sure you have the correct version of OpenCV installed for your platform.

4. **Flask app crashes**: Check the terminal for error messages. Common issues include permission problems with camera access or memory limitations.