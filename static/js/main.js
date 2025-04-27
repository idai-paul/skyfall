/**
 * Smart City Surveillance System - Main JavaScript
 * 
 * This script handles dynamic functionality of the application
 * including real-time updates, user interactions, and animations.
 */

// Global variables
let map;
let markers = [];
let currentCamera = null;
let videoStreams = {};
let streamManager = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing Skyfall Platform...');
    
    // Initialize the map
    initMap();
    
    // Initialize the stream manager
    initStreamManager();
    
    // Set up event listeners
    setupEventListeners();
    
    // Load initial data
    loadInitialData();
});

// Initialize the map
function initMap() {
    console.log('Initializing map...');
    
    // Create the map
    map = L.map('map').setView([51.505, -0.09], 13);
    
    // Add the tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);
    
    console.log('Map initialized');
}

// Initialize the stream manager
function initStreamManager() {
    console.log('Initializing stream manager...');
    
    // Create the stream manager
    streamManager = new StreamManager();
    
    // Set up event listeners for the stream manager
    streamManager.on('streamStarted', handleStreamStarted);
    streamManager.on('streamStopped', handleStreamStopped);
    streamManager.on('streamError', handleStreamError);
    
    console.log('Stream manager initialized');
}

// Set up event listeners
function setupEventListeners() {
    console.log('Setting up event listeners...');
    
    // Camera selection
    document.querySelectorAll('.camera-select').forEach(select => {
        select.addEventListener('change', handleCameraSelection);
    });
    
    // Stream control buttons
    document.querySelectorAll('.stream-control').forEach(button => {
        button.addEventListener('click', handleStreamControl);
    });
    
    console.log('Event listeners set up');
}

// Load initial data
function loadInitialData() {
    console.log('Loading initial data...');
    
    // Load cameras
    loadCameras();
    
    // Load detections
    loadDetections();
    
    console.log('Initial data loaded');
}

// Load cameras
function loadCameras() {
    console.log('Loading cameras...');
    
    // Fetch cameras from the API
    fetch('/api/cameras')
        .then(response => response.json())
        .then(data => {
            console.log('Cameras loaded:', data);
            
            // Update the camera list
            updateCameraList(data);
            
            // Add markers to the map
            addCameraMarkers(data);
        })
        .catch(error => {
            console.error('Error loading cameras:', error);
        });
}

// Update the camera list
function updateCameraList(cameras) {
    console.log('Updating camera list...');
    
    // Get the camera list container
    const cameraList = document.getElementById('cameraList');
    
    // Clear the camera list
    cameraList.innerHTML = '';
    
    // Add each camera to the list
    cameras.forEach(camera => {
        // Create the camera item
        const cameraItem = document.createElement('div');
        cameraItem.className = 'camera-item';
        cameraItem.dataset.cameraId = camera.id;
        
        // Create the camera name
        const cameraName = document.createElement('div');
        cameraName.className = 'camera-name';
        cameraName.textContent = camera.name;
        
        // Create the camera status
        const cameraStatus = document.createElement('div');
        cameraStatus.className = 'camera-status';
        
        // Create the status indicator
        const statusIndicator = document.createElement('div');
        statusIndicator.className = 'status-indicator';
        statusIndicator.classList.add(camera.status === 'running' ? 'online' : 'offline');
        
        // Create the status text
        const statusText = document.createElement('span');
        statusText.textContent = camera.status === 'running' ? 'Online' : 'Offline';
        
        // Add the status indicator and text to the camera status
        cameraStatus.appendChild(statusIndicator);
        cameraStatus.appendChild(statusText);
        
        // Add the camera name and status to the camera item
        cameraItem.appendChild(cameraName);
        cameraItem.appendChild(cameraStatus);
        
        // Add the camera item to the camera list
        cameraList.appendChild(cameraItem);
        
        // Add click event listener to the camera item
        cameraItem.addEventListener('click', () => {
            selectCamera(camera);
        });
    });
    
    console.log('Camera list updated');
}

// Add camera markers to the map
function addCameraMarkers(cameras) {
    console.log('Adding camera markers to the map...');
    
    // Clear existing markers
    markers.forEach(marker => marker.remove());
    markers = [];
    
    // Add a marker for each camera
    cameras.forEach(camera => {
        // Create the marker
        const marker = L.marker([camera.latitude, camera.longitude])
            .bindPopup(`<b>${camera.name}</b><br>Status: ${camera.status}`)
            .addTo(map);
        
        // Add the marker to the markers array
        markers.push(marker);
        
        // Add click event listener to the marker
        marker.on('click', () => {
            selectCamera(camera);
        });
    });
    
    console.log('Camera markers added to the map');
}

// Select a camera
function selectCamera(camera) {
    console.log('Selecting camera:', camera);
    
    // Update the current camera
    currentCamera = camera;
    
    // Update the UI
    updateCameraUI(camera);
    
    // Start the stream if the camera is running
    if (camera.status === 'running') {
        startStream(camera.id);
    }
}

// Update the camera UI
function updateCameraUI(camera) {
    console.log('Updating camera UI:', camera);
    
    // Update the camera name
    document.getElementById('cameraName').textContent = camera.name;
    
    // Update the camera status
    document.getElementById('cameraStatus').textContent = camera.status === 'running' ? 'Online' : 'Offline';
    
    // Update the camera location
    document.getElementById('cameraLocation').textContent = `${camera.latitude}, ${camera.longitude}`;
    
    // Update the stream control buttons
    updateStreamControlButtons(camera);
    
    // Center the map on the camera
    map.setView([camera.latitude, camera.longitude], 15);
    
    // Highlight the selected camera in the list
    document.querySelectorAll('.camera-item').forEach(item => {
        if (item.dataset.cameraId === camera.id) {
            item.classList.add('selected');
        } else {
            item.classList.remove('selected');
        }
    });
}

// Update the stream control buttons
function updateStreamControlButtons(camera) {
    console.log('Updating stream control buttons:', camera);
    
    // Get the stream control buttons
    const startButton = document.getElementById('startStream');
    const stopButton = document.getElementById('stopStream');
    
    // Update the button states
    if (camera.status === 'running') {
        startButton.disabled = true;
        stopButton.disabled = false;
    } else {
        startButton.disabled = false;
        stopButton.disabled = true;
    }
}

// Handle camera selection
function handleCameraSelection(event) {
    console.log('Camera selection changed:', event.target.value);
    
    // Get the selected camera
    const cameraId = event.target.value;
    
    // Find the camera in the cameras array
    const camera = cameras.find(c => c.id === cameraId);
    
    // Select the camera
    if (camera) {
        selectCamera(camera);
    }
}

// Handle stream control
function handleStreamControl(event) {
    console.log('Stream control clicked:', event.target.id);
    
    // Get the camera ID
    const cameraId = currentCamera.id;
    
    // Start or stop the stream
    if (event.target.id === 'startStream') {
        startStream(cameraId);
    } else if (event.target.id === 'stopStream') {
        stopStream(cameraId);
    }
}

// Start a stream
function startStream(cameraId) {
    console.log('Starting stream for camera:', cameraId);
    
    // Start the stream
    streamManager.startStream(cameraId);
}

// Stop a stream
function stopStream(cameraId) {
    console.log('Stopping stream for camera:', cameraId);
    
    // Stop the stream
    streamManager.stopStream(cameraId);
}

// Handle stream started
function handleStreamStarted(cameraId) {
    console.log('Stream started for camera:', cameraId);
    
    // Update the UI
    updateStreamUI(cameraId, true);
}

// Handle stream stopped
function handleStreamStopped(cameraId) {
    console.log('Stream stopped for camera:', cameraId);
    
    // Update the UI
    updateStreamUI(cameraId, false);
}

// Handle stream error
function handleStreamError(cameraId, error) {
    console.error('Stream error for camera:', cameraId, error);
    
    // Show an error message
    showErrorMessage(`Error streaming from camera ${cameraId}: ${error.message}`);
}

// Update the stream UI
function updateStreamUI(cameraId, isStreaming) {
    console.log('Updating stream UI for camera:', cameraId, 'isStreaming:', isStreaming);
    
    // Get the video container
    const videoContainer = document.getElementById('videoContainer');
    
    // Get the loading indicator
    const loadingIndicator = document.getElementById('loadingIndicator');
    
    // Get the status indicator
    const statusIndicator = document.getElementById('statusIndicator');
    
    // Update the UI based on the streaming state
    if (isStreaming) {
        // Hide the loading indicator
        loadingIndicator.style.display = 'none';
        
        // Show the status indicator
        statusIndicator.style.display = 'block';
        
        // Update the video source
        const video = document.getElementById('cameraFeed');
        video.src = `/api/streams/${cameraId}`;
    } else {
        // Show the loading indicator
        loadingIndicator.style.display = 'block';
        
        // Hide the status indicator
        statusIndicator.style.display = 'none';
        
        // Clear the video source
        const video = document.getElementById('cameraFeed');
        video.src = '';
    }
}

// Show an error message
function showErrorMessage(message) {
    console.error('Error:', message);
    
    // Create the error message element
    const errorMessage = document.createElement('div');
    errorMessage.className = 'error-message';
    errorMessage.textContent = message;
    
    // Add the error message to the page
    document.body.appendChild(errorMessage);
    
    // Remove the error message after 5 seconds
    setTimeout(() => {
        errorMessage.remove();
    }, 5000);
}

// Load detections
function loadDetections() {
    console.log('Loading detections...');
    
    // Fetch detections from the API
    fetch('/api/detections')
        .then(response => response.json())
        .then(data => {
            console.log('Detections loaded:', data);
            
            // Update the detections list
            updateDetectionsList(data);
        })
        .catch(error => {
            console.error('Error loading detections:', error);
        });
}

// Update the detections list
function updateDetectionsList(detections) {
    console.log('Updating detections list...');
    
    // Get the detections list container
    const detectionsList = document.getElementById('detectionsList');
    
    // Clear the detections list
    detectionsList.innerHTML = '';
    
    // Add each detection to the list
    detections.forEach(detection => {
        // Create the detection item
        const detectionItem = document.createElement('div');
        detectionItem.className = 'detection-item';
        
        // Create the detection time
        const detectionTime = document.createElement('div');
        detectionTime.className = 'detection-time';
        detectionTime.textContent = new Date(detection.timestamp).toLocaleString();
        
        // Create the detection type
        const detectionType = document.createElement('div');
        detectionType.className = 'detection-type';
        detectionType.textContent = detection.type;
        
        // Create the detection confidence
        const detectionConfidence = document.createElement('div');
        detectionConfidence.className = 'detection-confidence';
        detectionConfidence.textContent = `${(detection.confidence * 100).toFixed(1)}%`;
        
        // Add the detection time, type, and confidence to the detection item
        detectionItem.appendChild(detectionTime);
        detectionItem.appendChild(detectionType);
        detectionItem.appendChild(detectionConfidence);
        
        // Add the detection item to the detections list
        detectionsList.appendChild(detectionItem);
    });
    
    console.log('Detections list updated');
}

// Stream Manager class
class StreamManager {
    constructor() {
        this.streams = {};
        this.eventListeners = {};
    }
    
    // Start a stream
    startStream(cameraId) {
        console.log('StreamManager: Starting stream for camera:', cameraId);
        
        // Check if the stream is already running
        if (this.streams[cameraId]) {
            console.log('StreamManager: Stream already running for camera:', cameraId);
            return;
        }
        
        // Create a new stream
        this.streams[cameraId] = {
            id: cameraId,
            status: 'starting',
            error: null
        };
        
        // Emit the streamStarted event
        this.emit('streamStarted', cameraId);
    }
    
    // Stop a stream
    stopStream(cameraId) {
        console.log('StreamManager: Stopping stream for camera:', cameraId);
        
        // Check if the stream is running
        if (!this.streams[cameraId]) {
            console.log('StreamManager: Stream not running for camera:', cameraId);
            return;
        }
        
        // Update the stream status
        this.streams[cameraId].status = 'stopping';
        
        // Emit the streamStopped event
        this.emit('streamStopped', cameraId);
        
        // Remove the stream
        delete this.streams[cameraId];
    }
    
    // Add an event listener
    on(event, callback) {
        console.log('StreamManager: Adding event listener for event:', event);
        
        // Check if the event exists
        if (!this.eventListeners[event]) {
            this.eventListeners[event] = [];
        }
        
        // Add the callback to the event listeners
        this.eventListeners[event].push(callback);
    }
    
    // Remove an event listener
    off(event, callback) {
        console.log('StreamManager: Removing event listener for event:', event);
        
        // Check if the event exists
        if (!this.eventListeners[event]) {
            return;
        }
        
        // Remove the callback from the event listeners
        this.eventListeners[event] = this.eventListeners[event].filter(cb => cb !== callback);
    }
    
    // Emit an event
    emit(event, ...args) {
        console.log('StreamManager: Emitting event:', event, 'with args:', args);
        
        // Check if the event exists
        if (!this.eventListeners[event]) {
            return;
        }
        
        // Call each callback with the arguments
        this.eventListeners[event].forEach(callback => {
            callback(...args);
        });
    }
}