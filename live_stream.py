"""
Live Stream Clipper Module

This module provides a StreamClipper class that creates short clips from
YouTube live streams using ffmpeg and yt-dlp.
"""

import os
import time
import threading
import sys
import subprocess
import logging
from datetime import datetime
import yt_dlp
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class StreamClipper:
    def __init__(self, video_id, output_dir="youtube_clips", clip_duration=5, max_attempts=3, cooldown=2, no_signals=False):
        """
        Initialize a stream clipper
        
        Args:
            video_id: YouTube video ID
            output_dir: Directory to store clips
            clip_duration: Duration of each clip in seconds
            max_attempts: Maximum number of attempts for each clip
            cooldown: Cooldown time between attempts in seconds
            no_signals: If True, disable signal handling (for thread safety)
        """
        self.video_id = video_id
        self.output_dir = output_dir
        self.clip_duration = clip_duration
        self.max_attempts = max_attempts
        self.cooldown = cooldown
        self.no_signals = no_signals
        
        self.is_running = False
        self.clip_thread = None
        self.last_url = None
        self.url_fetch_time = 0
        self.url_valid_duration = 120  # Consider URL valid for 2 minutes
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up signal handling only in the main thread if not disabled
        # This is the key change to fix the signal error
        if not no_signals and threading.current_thread() is threading.main_thread():
            try:
                import signal
                signal.signal(signal.SIGINT, self._signal_handler)
                logger.info("Signal handling set up")
            except Exception as e:
                logger.warning(f"Could not set up signal handling: {e}")
    
    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C and other signals"""
        logger.info("Shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def _get_stream_url(self, force_refresh=False):
        """Get the direct stream URL using yt-dlp"""
        current_time = time.time()
        
        # Return cached URL if it's still valid
        if (not force_refresh and 
            self.last_url and 
            current_time - self.url_fetch_time < self.url_valid_duration):
            return self.last_url
        
        try:
            logger.info(f"Getting stream URL for video: {self.video_id}")
            youtube_url = f"https://www.youtube.com/watch?v={self.video_id}"
            
            # Use a specific format to ensure we get video and audio
            ydl_opts = {
                'format': 'best[height<=720]/bestvideo[height<=720]+bestaudio',
                'quiet': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                if 'url' in info:
                    logger.info("Successfully retrieved stream URL")
                    self.last_url = info['url']
                    self.url_fetch_time = current_time
                    return self.last_url
                else:
                    logger.error("No URL found in video info")
                    return None
        except Exception as e:
            logger.error(f"Error getting stream URL: {e}")
            return None
    
    def _create_clip(self, timestamp=None):
        """Create a single clip"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_file = os.path.join(self.output_dir, f"clip_{timestamp}.mp4")
        
        # Try multiple times in case of URL expiration
        for attempt in range(self.max_attempts):
            # Get fresh URL for each attempt if previous failed
            force_refresh = attempt > 0
            stream_url = self._get_stream_url(force_refresh=force_refresh)
            
            if not stream_url:
                logger.error(f"Failed to get stream URL (attempt {attempt+1}/{self.max_attempts})")
                time.sleep(self.cooldown)
                continue
            
            # FFmpeg command to record a clip
            cmd = [
                'ffmpeg',
                '-y',                      # Overwrite output files
                '-i', stream_url,          # Input stream
                '-t', str(self.clip_duration),  # Duration
                '-c:v', 'libx264',         # Video codec
                '-preset', 'ultrafast',    # Encoding speed
                '-c:a', 'aac',             # Audio codec
                '-b:a', '128k',            # Audio bitrate
                '-loglevel', 'warning',    # Reduce logging
                '-movflags', '+faststart', # Optimize for streaming
                output_file                # Output file
            ]
            
            logger.info(f"Creating clip: {os.path.basename(output_file)}")
            
            try:
                # Run FFmpeg process with timeout
                process = subprocess.run(
                    cmd, 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=self.clip_duration * 2  # Reasonable timeout
                )
                
                # Check if successful
                if process.returncode == 0:
                    # Verify file exists and has size
                    if os.path.exists(output_file) and os.path.getsize(output_file) > 10000:  # > 10KB
                        file_size = os.path.getsize(output_file) / 1024  # KB
                        logger.info(f"Clip created: {os.path.basename(output_file)} ({file_size:.1f} KB)")
                        return True
                    else:
                        logger.warning(f"Clip file is too small or missing (attempt {attempt+1}/{self.max_attempts})")
                else:
                    logger.warning(f"FFmpeg failed with code {process.returncode} (attempt {attempt+1}/{self.max_attempts})")
                    stderr = process.stderr.decode('utf-8', errors='ignore')
                    if "403 Forbidden" in stderr or "Invalid argument" in stderr:
                        # URL likely expired, force refresh on next attempt
                        self.last_url = None
                
            except subprocess.TimeoutExpired:
                logger.warning(f"FFmpeg timed out (attempt {attempt+1}/{self.max_attempts})")
                # Kill the process if it's still running
                try:
                    process.kill()
                except:
                    pass
            
            # Wait before retrying
            time.sleep(self.cooldown)
        
        logger.error(f"Failed to create clip after {self.max_attempts} attempts")
        return False
    
    def _clip_loop(self):
        """Create clips continuously"""
        while self.is_running:
            try:
                # Get current timestamp for clip name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Create a clip
                success = self._create_clip(timestamp)
                
                # Add a small random delay between clips (1-3 seconds)
                # This helps avoid exact timing patterns that might trigger rate limits
                delay = self.cooldown + random.uniform(1, 3)
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error in clip loop: {e}")
                time.sleep(5)  # Longer delay on unexpected errors
    
    def start(self):
        """Start creating clips"""
        if self.is_running:
            logger.warning("Already running")
            return
        
        self.is_running = True
        
        # Start clip creation thread
        self.clip_thread = threading.Thread(target=self._clip_loop)
        self.clip_thread.daemon = True
        self.clip_thread.start()
        
        logger.info(f"Started creating {self.clip_duration}s clips in {self.output_dir}")
    
    def stop(self):
        """Stop creating clips"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Wait for clip thread to finish
        if self.clip_thread and self.clip_thread.is_alive():
            logger.info("Waiting for clip thread to finish...")
            self.clip_thread.join(timeout=10)
        
        logger.info("Clipper stopped")


# Main function
def main():
    video_id = "t4Hl35oF7Dg"  # Castro Street Cam
    
    clipper = StreamClipper(
        video_id=video_id,
        output_dir=f"youtube_clips/{video_id}",
        clip_duration=10,
        max_attempts=3,
        cooldown=2
    )
    
    try:
        clipper.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Stopping due to keyboard interrupt...")
    finally:
        clipper.stop()

if __name__ == "__main__":
    main()