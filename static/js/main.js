/**
 * Smart City Surveillance System - Main JavaScript
 * 
 * This script handles dynamic functionality of the application
 * including real-time updates, user interactions, and animations.
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize real-time clock
    initializeClock();
    
    // Initialize subjects clicking
    initializeSubjectSelection();
    
    // For demonstration purposes, simulate camera feeds
    simulateCameraFeeds();
    
    // Simulate detection tracking
    simulateDetectionTracking();
    
    // Simulate feature point blinking
    simulateFeaturePointActivity();
});

/**
 * Initialize the real-time clock in the header
 */
function initializeClock() {
    const clockElement = document.getElementById('current-time');
    
    function updateClock() {
        const now = new Date();
        const timeString = now.toLocaleTimeString();
        clockElement.textContent = timeString;
    }
    
    // Update clock immediately and then every second
    updateClock();
    setInterval(updateClock, 1000);
}

/**
 * Initialize subject selection functionality
 */
function initializeSubjectSelection() {
    const subjectCards = document.querySelectorAll('.subject-card');
    const profileSection = document.querySelector('.profile-details');
    
    subjectCards.forEach(card => {
        card.addEventListener('click', function() {
            // Remove active class from all cards
            subjectCards.forEach(c => c.querySelector('.card').classList.remove('border-primary'));
            
            // Add active class to clicked card
            this.querySelector('.card').classList.add('border-primary');
            
            // Get subject ID
            const subjectId = this.getAttribute('data-subject-id');
            
            // In a real application, this would fetch subject details
            // For now, we'll just highlight the card with a brief animation
            this.querySelector('.card').style.transform = 'scale(1.05)';
            setTimeout(() => {
                this.querySelector('.card').style.transform = 'scale(1)';
            }, 200);
            
            // Update profile section (in a real app, this would load actual data)
            // Here we're just simulating that the profile changed
            profileSection.style.opacity = '0.5';
            setTimeout(() => {
                // If we clicked on the second subject, update the profile to match
                if(subjectId === "19A7") {
                    profileSection.querySelector('h3').textContent = "Subject #19A7";
                    profileSection.querySelector('h3').className = "h6 text-primary";
                    profileSection.querySelector('p').textContent = "Match Confidence: 87%";
                    
                    // Update other profile details
                    const detailValues = profileSection.querySelectorAll('.detail-value');
                    detailValues[0].textContent = "14:12:55"; // First seen
                    detailValues[1].textContent = "3"; // Appearances
                    detailValues[2].textContent = "Market Sq"; // Location
                    detailValues[3].textContent = "25-35"; // Age
                    detailValues[4].textContent = "Female"; // Gender
                    detailValues[5].textContent = "5'6\" (168cm)"; // Height
                    detailValues[6].textContent = "Tentative"; // Status
                    
                    // Update features
                    const featureTags = profileSection.querySelector('.feature-tags');
                    featureTags.innerHTML = `
                        <span class="feature-tag">Red coat</span>
                        <span class="feature-tag">Backpack</span>
                        <span class="feature-tag">Hat</span>
                        <span class="feature-tag">Boots</span>
                    `;
                } else {
                    // Default back to subject #28F4
                    profileSection.querySelector('h3').textContent = "Subject #28F4";
                    profileSection.querySelector('h3').className = "h6 text-danger";
                    profileSection.querySelector('p').textContent = "Match Confidence: 92%";
                    
                    // Reset other profile details
                    const detailValues = profileSection.querySelectorAll('.detail-value');
                    detailValues[0].textContent = "14:17:22"; // First seen
                    detailValues[1].textContent = "5"; // Appearances
                    detailValues[2].textContent = "Main Street"; // Location
                    detailValues[3].textContent = "35-45"; // Age
                    detailValues[4].textContent = "Male"; // Gender
                    detailValues[5].textContent = "5'10\" (178cm)"; // Height
                    detailValues[6].textContent = "Confirmed"; // Status
                    
                    // Reset features
                    const featureTags = profileSection.querySelector('.feature-tags');
                    featureTags.innerHTML = `
                        <span class="feature-tag">Black jacket</span>
                        <span class="feature-tag">Blue cap</span>
                        <span class="feature-tag">Glasses</span>
                        <span class="feature-tag">Beard</span>
                    `;
                }
                
                profileSection.style.opacity = '1';
            }, 300);
        });
    });
}

/**
 * Simulate activity on camera feeds for demonstration
 */
function simulateCameraFeeds() {
    const cameraFeeds = document.querySelectorAll('.camera-feed-placeholder');
    
    cameraFeeds.forEach(feed => {
        // Create a simulated activity indicator
        const activityIndicator = document.createElement('div');
        activityIndicator.classList.add('activity-indicator');
        activityIndicator.style.position = 'absolute';
        activityIndicator.style.bottom = '5px';
        activityIndicator.style.right = '5px';
        activityIndicator.style.width = '6px';
        activityIndicator.style.height = '6px';
        activityIndicator.style.borderRadius = '50%';
        activityIndicator.style.backgroundColor = '#00FF00';
        
        feed.appendChild(activityIndicator);
        
        // Simulate video availability toggle with button
        const toggleButton = document.createElement('button');
        toggleButton.classList.add('btn', 'btn-sm', 'btn-outline-secondary', 'position-absolute');
        toggleButton.style.top = '5px';
        toggleButton.style.right = '5px';
        toggleButton.style.fontSize = '0.7rem';
        toggleButton.style.padding = '2px 5px';
        toggleButton.textContent = 'Toggle Feed';
        
        toggleButton.addEventListener('click', function(e) {
            e.stopPropagation(); // Prevent event from bubbling up
            
            const placeholderText = feed.querySelector('.placeholder-text');
            if(placeholderText.textContent.includes('Unavailable')) {
                placeholderText.innerHTML = `
                    <i class="fas fa-check-circle placeholder-icon text-success"></i>
                    <span>Feed Active (Simulated)</span>
                `;
            } else {
                placeholderText.innerHTML = `
                    <i class="fas fa-video placeholder-icon"></i>
                    <span>Feed Unavailable</span>
                `;
            }
        });
        
        feed.appendChild(toggleButton);
    });
}

/**
 * Simulate detection box movement for demonstration
 */
function simulateDetectionTracking() {
    const detectionBoxes = document.querySelectorAll('.detection-box');
    
    detectionBoxes.forEach(box => {
        // Initial position - centered by default
        let posX = 0;
        let posY = 0;
        let dirX = Math.random() > 0.5 ? 1 : -1;
        let dirY = Math.random() > 0.5 ? 1 : -1;
        
        function moveBox() {
            // Change direction occasionally
            if (Math.random() < 0.05) {
                dirX = -dirX;
            }
            if (Math.random() < 0.05) {
                dirY = -dirY;
            }
            
            // Update position
            posX += dirX * 0.3;
            posY += dirY * 0.2;
            
            // Keep within bounds
            if (posX > 15 || posX < -15) {
                dirX = -dirX;
                posX = posX > 0 ? 15 : -15;
            }
            if (posY > 10 || posY < -10) {
                dirY = -dirY;
                posY = posY > 0 ? 10 : -10;
            }
            
            // Apply transform - add to the existing translation for the centering
            box.style.transform = `translateX(calc(-50% + ${posX}px)) translateY(${posY}px)`;
            
            // Continue animation
            requestAnimationFrame(moveBox);
        }
        
        // Start animation
        moveBox();
    });
}

/**
 * Simulate feature point blinking and activity
 */
function simulateFeaturePointActivity() {
    const featurePoints = document.querySelectorAll('.feature-point');
    
    featurePoints.forEach(point => {
        // Simulate activity with random blinking
        setInterval(() => {
            // Only blink sometimes
            if (Math.random() < 0.3) {
                point.style.opacity = '0.3';
                setTimeout(() => {
                    point.style.opacity = '1';
                }, 200);
            }
        }, 2000 * Math.random() + 1000); // Random interval between 1-3 seconds
    });
}

/**
 * Function to refresh all feeds - simulates live updating
 */
function refreshFeeds() {
    // Simulate some activity by updating timestamps
    const timestamps = document.querySelectorAll('.timestamp');
    timestamps.forEach(timestamp => {
        const hours = Math.floor(Math.random() * 24).toString().padStart(2, '0');
        const minutes = Math.floor(Math.random() * 60).toString().padStart(2, '0');
        const seconds = Math.floor(Math.random() * 60).toString().padStart(2, '0');
        timestamp.textContent = `${hours}:${minutes}:${seconds}`;
    });
    
    // Simulate path activity - make current location markers pulse
    const pathMarkers = document.querySelectorAll('.path-current-location');
    pathMarkers.forEach(marker => {
        marker.style.transform = 'translate(-50%, -50%) scale(1.5)';
        setTimeout(() => {
            marker.style.transform = 'translate(-50%, -50%) scale(1)';
        }, 200);
    });
}

// Add action button functionality
document.addEventListener('DOMContentLoaded', function() {
    // Simulate action button clicks
    const actionButtons = document.querySelectorAll('.action-buttons button');
    
    actionButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Visual feedback
            button.disabled = true;
            button.textContent = 'Processing...';
            
            setTimeout(() => {
                // Simulate completion
                button.textContent = 'Completed';
                button.classList.remove('btn-secondary');
                button.classList.add('btn-success');
                
                // Reset after a few seconds
                setTimeout(() => {
                    button.disabled = false;
                    button.classList.remove('btn-success');
                    button.classList.add('btn-secondary');
                    button.textContent = button.textContent === 'Completed' ? 
                        (button.getAttribute('data-original-text') || 'Action') : button.textContent;
                }, 3000);
            }, 1500);
        });
        
        // Store original text
        button.setAttribute('data-original-text', button.textContent);
    });
});

// Set up a simple refresh interval
setInterval(refreshFeeds, 30000); // Refresh every 30 seconds

// Function to update detection boxes
function updateDetectionBoxes(cameraId, detections) {
    const detectionBoxesContainer = document.querySelector(`#camera-${cameraId} .detection-boxes`);
    if (!detectionBoxesContainer) return;

    // Clear existing boxes
    detectionBoxesContainer.innerHTML = '';

    // Add new boxes
    detections.forEach(detection => {
        const box = document.createElement('div');
        box.className = 'detection-box';
        box.style.borderColor = detection.color;
        box.style.left = `${detection.x}%`;
        box.style.top = `${detection.y}%`;
        box.style.width = `${detection.width}%`;
        box.style.height = `${detection.height}%`;

        const label = document.createElement('div');
        label.className = 'subject-id';
        label.textContent = `Subject ${detection.subject_id}`;
        label.style.color = detection.color;

        box.appendChild(label);
        detectionBoxesContainer.appendChild(box);
    });
}

// WebSocket connection for real-time updates
let ws = null;

function connectWebSocket() {
    ws = new WebSocket(`ws://${window.location.host}/ws`);

    ws.onopen = () => {
        console.log('WebSocket connection established');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'detection') {
            updateDetectionBoxes(data.camera_id, data.detections);
        }
    };

    ws.onclose = () => {
        console.log('WebSocket connection closed. Reconnecting...');
        setTimeout(connectWebSocket, 1000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

// Connect to WebSocket when the page loads
document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
});

// Handle video stream errors
document.querySelectorAll('video').forEach(video => {
    video.addEventListener('error', (e) => {
        console.error('Video stream error:', e);
        const container = video.closest('.video-container');
        if (container) {
            container.innerHTML = '<div class="alert alert-danger">Failed to load video stream</div>';
        }
    });
});