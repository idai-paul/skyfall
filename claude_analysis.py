#!/usr/bin/env python3
"""
Person Analyzer - Analyzes full-body images of people using Claude API

This script analyzes a directory of full-body images using Anthropic's Claude API
to identify clothing, accessories, and other details about people in the images.
It outputs the analysis in a structured JSON format.

Requirements:
- anthropic package
- Pillow (PIL) package
- A valid Anthropic API key
"""

import os
import json
import base64
import argparse
from typing import Dict, List, Any, Optional
import time
import glob
from pathlib import Path

# Import required libraries - make sure to install these first
import anthropic
from PIL import Image
import io

class PersonAnalyzer:
    """Analyzes images of people using Anthropic's Claude API"""
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        """
        Initialize the PersonAnalyzer
        
        Args:
            api_key: Anthropic API key
            model: Model to use for analysis (default: claude-3-opus-20240229)
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode an image as base64
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def analyze_image(self, image_path: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Analyze a single image using Claude API
        
        Args:
            image_path: Path to the image file
            max_retries: Maximum number of retries for API call
            
        Returns:
            Dictionary with analysis results
        """
        # Extract person ID from filename (assuming format "person_ID_full.jpg")
        person_id = os.path.basename(image_path).split('_')[1]
        
        # Encode image to base64
        try:
            base64_image = self.encode_image(image_path)
        except Exception as e:
            print(f"Error encoding image {image_path}: {str(e)}")
            return {
                "person_id": person_id,
                "error": f"Failed to encode image: {str(e)}",
                "path": image_path
            }
        
        # Set up the prompt
        prompt = """
        Please analyze this image of a person and provide a detailed description focusing on:
        
        1. General appearance (age range, gender if apparent, height/build)
        2. Clothing (top, bottom, outerwear, footwear)
        3. Accessories (bags, hats, glasses, jewelry, etc.)
        4. Any objects the person is carrying or interacting with
        5. Any notable posture, stance, or movement
        
        Format your response as JSON with the following structure:
        {
            "appearance": {
                "estimated_age_range": "",
                "gender": "",
                "build": ""
            },
            "clothing": {
                "top": "",
                "bottom": "",
                "outerwear": "",
                "footwear": ""
            },
            "accessories": [
                {"type": "", "description": ""}
            ],
            "carried_objects": [
                {"type": "", "description": ""}
            ],
            "posture_movement": ""
        }
        
        Only include fields if you can determine them with reasonable confidence.
        """
        
        # Make API request with retries
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": base64_image
                                    }
                                }
                            ]
                        }
                    ]
                )
                
                # Extract JSON from response
                try:
                    # Try to parse the response text as JSON
                    response_text = response.content[0].text
                    
                    # Find JSON block in the text if it's not pure JSON
                    if not response_text.strip().startswith('{'):
                        import re
                        json_match = re.search(r'({[\s\S]*})', response_text)
                        if json_match:
                            response_text = json_match.group(1)
                    
                    analysis = json.loads(response_text)
                    
                    # Add metadata
                    result = {
                        "person_id": person_id,
                        "path": image_path,
                        "analysis": analysis
                    }
                    
                    return result
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON from response for {image_path}: {str(e)}")
                    print(f"Response content: {response.content[0].text}")
                    
                    # Return raw text if JSON parsing fails
                    return {
                        "person_id": person_id,
                        "path": image_path,
                        "error": "Failed to parse JSON response",
                        "raw_response": response.content[0].text
                    }
                    
            except Exception as e:
                print(f"Attempt {attempt+1}/{max_retries} failed for {image_path}: {str(e)}")
                if attempt < max_retries - 1:
                    # Add exponential backoff
                    sleep_time = 2 ** attempt
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    return {
                        "person_id": person_id,
                        "path": image_path,
                        "error": f"API request failed after {max_retries} attempts: {str(e)}"
                    }
    
    def analyze_directory(self, directory_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze all images in a directory
        
        Args:
            directory_path: Path to directory containing person images
            output_path: Path to save JSON output (if None, won't save to disk)
            
        Returns:
            Dictionary mapping person IDs to analysis results
        """
        # Find all full-body images
        image_paths = glob.glob(os.path.join(directory_path, "person_*_full.jpg"))
        
        if not image_paths:
            print(f"No images matching 'person_*_full.jpg' found in {directory_path}")
            return {}
        
        print(f"Found {len(image_paths)} images to analyze")
        
        # Analyze each image
        results = {}
        for i, image_path in enumerate(image_paths):
            print(f"Analyzing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            analysis = self.analyze_image(image_path)
            
            # Store by person ID
            person_id = analysis["person_id"]
            results[person_id] = analysis
        
        # Save results to JSON file if output path provided
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Analysis saved to {output_path}")
        
        return results


def main():
    """Main function to run the analyzer from command line"""
    parser = argparse.ArgumentParser(description="Analyze people in images using Claude API")
    parser.add_argument("--dir", required=True, help="Directory containing person_*_full.jpg images")
    parser.add_argument("--output", default="person_analysis.json", help="Output JSON file path")
    parser.add_argument("--api-key", help="Anthropic API key (if not provided, will check ANTHROPIC_API_KEY env var)")
    parser.add_argument("--model", default="claude-3-opus-20240229", help="Claude model to use")
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("API key must be provided via --api-key or ANTHROPIC_API_KEY environment variable")
    
    # Initialize analyzer
    analyzer = PersonAnalyzer(api_key=api_key, model=args.model)
    
    # Run analysis
    results = analyzer.analyze_directory(args.dir, args.output)
    
    # Print summary
    print(f"Analysis complete. Processed {len(results)} images.")


if __name__ == "__main__":
    main()