import os
import logging
import base64
import json
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnvironmentConfig:
    """Configuration for API credentials and settings."""
    openai_api_key: str
    model_name: str = "gpt-4o"
    
    @classmethod
    def from_env(cls) -> 'EnvironmentConfig':
        """Create configuration from environment variables."""
        return cls(
            openai_api_key=os.getenv('OPENAI_API_KEY', ''),
        )
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not provided")

class ObjectDetector:
    """Handles object detection in images using OpenAI Vision."""
    
    def __init__(self, config: EnvironmentConfig):
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.model_name
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def detect_objects(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        Detect objects in an image with retry logic for API calls.
        Returns a list of objects with their confidence scores.
        
        Returns:
            List[Dict[str, Any]]: List of detected objects in the format:
                [{"name": str, "confidence": float}, ...]
        """
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            prompt = """Analyze this image and list all visible objects. Return
the response in the following JSON format ONLY:

[
    {
        "name": "object name",
        "confidence": confidence_score
    },
    ...
]

Requirements:
1. The response must be valid JSON
2. Each confidence score must be a float between 0.0 and 1.0
3. Names should be simple, lowercase strings (e.g., "chair", "table", "person")
4. Do not include any explanation or additional text
5. The output should be an array of objects, even if only one object is detected

Example of valid response:
[
    {"name": "chair", "confidence": 0.95},
    {"name": "laptop", "confidence": 0.88}
]"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": ("You are a precise computer vision system that "
                                    "returns only valid JSON in the exact format "
                                    "requested.")
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096,
                temperature=0.3  # Lower temperature for more consistent formatting
            )
            
            try:
                content = response.choices[0].message.content.strip()
                # Remove any potential markdown code block markers
                content = content.replace("```json", "").replace("```", "").strip()
                detected_objects = json.loads(content)
                
                # Validate the response format
                if not isinstance(detected_objects, list):
                    raise ValueError("Response is not a list")
                
                for obj in detected_objects:
                    if not isinstance(obj, dict):
                        raise ValueError("Object is not a dictionary")
                    if "name" not in obj or "confidence" not in obj:
                        raise ValueError("Object missing required fields")
                    if not isinstance(obj["name"], str):
                        raise ValueError("Object name is not a string")
                    if not isinstance(obj["confidence"], (int, float)):
                        raise ValueError("Confidence score is not a number")
                    if not 0 <= obj["confidence"] <= 1:
                        raise ValueError("Confidence score out of range")
                
                logger.info(f"Successfully detected {len(detected_objects)} objects")
                return detected_objects
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {content}")
                raise
            except ValueError as e:
                logger.error(f"Invalid response format: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error detecting objects: {str(e)}")
            raise

class SafetyAnalyzer:
    """Generates safety recommendations using OpenAI's API."""
    
    def __init__(self, config: EnvironmentConfig):
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = "gpt-4o"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_recommendations(self, detected_objects: List[Dict[str, Any]]) -> str:
        """
        Generate safety recommendations with improved prompt engineering
        and structured output.
        """
        prompt = f"""Analyze the following detected objects and their
confidence scores:
{detected_objects}

Please provide:
1. Immediate safety concerns
2. Accessibility considerations
3. Recommended precautions

Format the response in clear, actionable bullet points focusing on the most
critical items first. Consider both physical safety and universal design principles.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ("You are a safety and accessibility expert "
                                                   "providing clear, practical advice.")},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4096
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise

def main(image_path: Path):
    """Main execution flow with proper error handling and logging."""
    try:
        # Initialize configuration
        config = EnvironmentConfig.from_env()
        config.validate()
        
        # Initialize services
        detector = ObjectDetector(config)
        analyzer = SafetyAnalyzer(config)
        
        # Process image
        logger.info(f"Processing image: {image_path}")
        detected_objects = detector.detect_objects(image_path)
        
        # Generate recommendations
        recommendations = analyzer.generate_recommendations(detected_objects)
        
        # Output results
        logger.info("Analysis complete")
        print("\nDetected Objects:")
        for obj in detected_objects:
            print(f"- {obj['name']} (Confidence: {obj['confidence']:.2f})")
        
        print("\nSafety Recommendations:")
        print(recommendations)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Get directory where script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Join it with your filename assuming it's in the same directory.
    # Replace filename if necessary.
    image_path = os.path.join(script_dir, "exercise_6.jpg")
    
    main(image_path)