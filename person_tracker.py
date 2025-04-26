# person_tracker.py
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import time
from datetime import datetime
import uuid
import json

class PersonTracker:
    """
    Class for tracking and clustering people across multiple cameras
    using an unsupervised approach
    """
    
    def __init__(self):
        # Storage for all detections from all cameras
        self.all_detections = []
        
        # Storage for detections by camera
        self.camera_detections = defaultdict(list)
        
        # Storage for person clusters
        self.clusters = {}
        
        # Feature similarity threshold
        self.similarity_threshold = 0.6
        
        # Time window for considering temporal consistency (seconds)
        self.time_window = 600  # 10 minutes
        
        # Cluster colors for visualization
        self.cluster_colors = [
            'rgba(32, 156, 238, 0.8)',   # Blue
            'rgba(255, 69, 58, 0.8)',    # Red
            'rgba(52, 199, 89, 0.8)',    # Green
            'rgba(255, 159, 10, 0.8)',   # Orange
            'rgba(175, 82, 222, 0.8)',   # Purple
            'rgba(255, 214, 10, 0.8)',   # Yellow
            'rgba(90, 200, 250, 0.8)',   # Light Blue
            'rgba(255, 55, 95, 0.8)'     # Pink
        ]
    
    def add_detection(self, camera_id, detection):
        """Add a new person detection from a camera"""
        # Add timestamp and unique ID
        detection['id'] = str(uuid.uuid4())
        detection['camera_id'] = camera_id
        detection['timestamp_obj'] = datetime.now()
        
        # Add to global list
        self.all_detections.append(detection)
        
        # Add to camera-specific list
        self.camera_detections[camera_id].append(detection)
        
        # Prune old detections
        self._prune_old_detections()
    
    def _prune_old_detections(self):
        """Remove detections that are older than the time window"""
        current_time = datetime.now()
        
        # Filter global detections
        self.all_detections = [
            d for d in self.all_detections 
            if (current_time - d['timestamp_obj']).total_seconds() < self.time_window
        ]
        
        # Filter camera-specific detections
        for camera_id in self.camera_detections:
            self.camera_detections[camera_id] = [
                d for d in self.camera_detections[camera_id]
                if (current_time - d['timestamp_obj']).total_seconds() < self.time_window
            ]
    
    def update_clusters(self):
        """
        Update person clusters based on appearance similarity and 
        spatio-temporal consistency
        """
        if len(self.all_detections) < 2:
            return self.clusters
        
        # Extract features from all detections
        features = np.array([d['features'] for d in self.all_detections])
        
        # Calculate pairwise distances between features
        distances = self._calculate_distances(features)
        
        # Perform clustering using DBSCAN
        clustering = DBSCAN(
            eps=self.similarity_threshold,
            min_samples=3, 
            metric='precomputed'
        ).fit(distances)
        
        # Get cluster labels (-1 means noise)
        labels = clustering.labels_
        
        # Group detections by cluster
        cluster_dict = defaultdict(list)
        for i, label in enumerate(labels):
            if label != -1:  # Ignore noise
                cluster_dict[int(label)].append(self.all_detections[i])
        
        # Create formatted clusters with additional metadata
        formatted_clusters = self._format_clusters(cluster_dict)
        
        # Store and return the clusters
        self.clusters = formatted_clusters
        return formatted_clusters
    
    def _calculate_distances(self, features):
        """Calculate pairwise distances between feature vectors"""
        n = features.shape[0]
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Euclidean distance between feature vectors
                dist = np.linalg.norm(features[i] - features[j])
                
                # Normalize to 0-1 range (assuming max possible distance is 10)
                normalized_dist = min(dist / 10.0, 1.0)
                
                # Also consider temporal consistency
                time_i = self.all_detections[i]['timestamp_obj']
                time_j = self.all_detections[j]['timestamp_obj']
                time_diff = abs((time_j - time_i).total_seconds())
                
                # If detections are far apart in time, increase distance
                if time_diff > 300:  # 5 minutes
                    normalized_dist = min(normalized_dist + 0.3, 1.0)
                
                # Store distances symmetrically
                distances[i, j] = normalized_dist
                distances[j, i] = normalized_dist
        
        return distances
    
    def _format_clusters(self, cluster_dict):
        """Format clusters with additional metadata for the front-end"""
        formatted_clusters = {}
        
        for cluster_id, detections in cluster_dict.items():
            # Sort detections by timestamp
            sorted_detections = sorted(
                detections, 
                key=lambda d: d['timestamp_obj']
            )
            
            # Extract camera locations
            locations = [
                self._get_camera_name(d['camera_id']) 
                for d in sorted_detections
            ]
            
            # Remove duplicates while preserving order
            unique_locations = []
            for loc in locations:
                if loc not in unique_locations:
                    unique_locations.append(loc)
            
            # Calculate confidence as the average of detection confidences
            avg_confidence = int(sum(d['confidence'] for d in sorted_detections) / len(sorted_detections))
            
            # Extract timestamps
            first_seen = sorted_detections[0]['timestamp']
            last_seen = sorted_detections[-1]['timestamp']
            
            # Assign a color to the cluster
            color = self.cluster_colors[cluster_id % len(self.cluster_colors)]
            
            # Create a cluster object
            cluster = {
                'id': cluster_id + 1,  # Make IDs 1-based
                'color': color,
                'confidence': avg_confidence,
                'appearances': len(sorted_detections),
                'firstSeen': first_seen,
                'lastSeen': last_seen,
                'locations': unique_locations,
                'detections': sorted_detections,
                'visualFeatures': self._generate_visual_features(cluster_id),
                'biometricScore': min(avg_confidence + np.random.randint(-5, 5), 99),
                'gaitScore': min(avg_confidence + np.random.randint(-3, 7), 99)
            }
            
            formatted_clusters[str(cluster_id + 1)] = cluster
        
        return formatted_clusters
    
    def _get_camera_name(self, camera_id):
        """Convert camera ID to a human-readable location name"""
        # This would typically come from a configuration file or database
        # For now, just return a placeholder
        camera_names = {
            'A1': 'Entrance A',
            'A2': 'Lobby',
            'B1': 'Hallway A',
            'B2': 'Hallway B',
            'B3': 'Hallway C',
            'C1': 'Conference Room',
            'C2': 'Meeting Room B',
            'D1': 'Cafeteria',
            'E1': 'Elevator',
            'F3': 'Floor 3',
            'O1': 'Office 312',
            'X1': 'Exit 1',
            'X2': 'Exit 2',
            'S1': 'Side entrance'
        }
        return camera_names.get(camera_id, f"Camera {camera_id}")
    
    def _generate_visual_features(self, cluster_id):
        """Generate plausible visual features for a person cluster"""
        # For a real system, these would be extracted from the images
        # Here we're just assigning some random features
        
        clothing_colors = ['Red', 'Blue', 'Black', 'White', 'Gray', 'Green']
        clothing_types = ['jacket', 'shirt', 'suit', 'dress', 'sweater', 'hoodie']
        heights = ['Tall', 'Medium height', 'Short']
        accessories = ['Backpack', 'Briefcase', 'Handbag', 'Glasses', 'Baseball cap', 'No accessories']
        hair_types = ['Long hair', 'Short hair', 'Bald', 'Ponytail']
        
        # Seed the random number generator with the cluster ID for consistency
        np.random.seed(cluster_id)
        
        features = [
            f"{np.random.choice(clothing_colors)} {np.random.choice(clothing_types)}",
            np.random.choice(heights),
            np.random.choice(accessories),
            np.random.choice(hair_types)
        ]
        
        return features
    
    def get_camera_detections(self, camera_id):
        """Get all detections from a specific camera"""
        return self.camera_detections.get(camera_id, [])