# orientation.py
import json
import base64
from PIL import Image
from io import BytesIO
import requests
import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Access the API key
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY not found in environment variables. Please set it in your .env file or in your environment.")

# Set up the Fireworks client
client = openai.Client(api_key=API_KEY, base_url="https://api.fireworks.ai/inference/v1")


def encode_image_base64(img):
    if img.mode in ('RGBA', 'LA'):
        img = img.convert('RGB')
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def encode_image_direct(image_path):
    with Image.open(image_path) as img:
        print(f"Loaded image: {image_path} (Size: {img.size}, Mode: {img.mode})")
        return encode_image_base64(img)

def get_orientation_from_llama(image_base64):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"

    system_prompt = (
        "You are a document validator. Your job is to ensure the document is readable. "
        "Make sure the text is not upside down or rotated incorrectly. If the text is upside down or at an angle, "
        "provide the correct orientation in degrees (0, 90, 180, 270) based on how a human would read it."
    )

    validation_prompt = "Give me the correct orientation of this document in degrees as a JSON object with the key 'orientation'."

    payload = {
        "model": "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
        "max_tokens": 1024,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": validation_prompt}
                ]
            }
        ]
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        response_json = response.json()
        print(f"\n--- Raw JSON response from Fireworks API ---\n{response_json}\n")

        raw_response = response_json.get("choices", [])[0].get("message", {}).get("content", "")
        parsed_json = json.loads(raw_response)
        
        orientation = parsed_json.get("orientation")

        if orientation is not None:
            print(f"Extracted orientation: {orientation} degrees")
            return int(orientation)
        else:
            print("Could not extract orientation.")
            return None

    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"API request or JSON parsing error: {e}")
        return None

def rotate_image(image, angle):
    print(f"Applying rotation of {angle} degrees to the image.")
    return image.rotate(-angle, expand=True)

def correct_image_orientation(image_path):
    with Image.open(image_path) as img:
        image_base64 = encode_image_base64(img)
        print(f"\nImage encoded as base64. Size: {len(image_base64)} characters.")

    orientation = get_orientation_from_llama(image_base64)

    if orientation is None:
        print("Failed to retrieve orientation.")
        return

    rotation_angle = 0
    if orientation == 90:
        rotation_angle = 90
    elif orientation == 180:
        rotation_angle = 180
    elif orientation == 270:
        rotation_angle = -90

    if rotation_angle != 0:
        print(f"Rotating image by {rotation_angle} degrees.")
        with Image.open(image_path) as img:
            rotated_image = rotate_image(img, rotation_angle)

            corrected_image_path = os.path.join("corrected_images", os.path.basename(image_path))
            save_image_with_quality(rotated_image, corrected_image_path)
            print(f"Corrected image saved at: {corrected_image_path}")
    else:
        print("Image is already in the correct orientation.")

def save_image_with_quality(image, output_path):
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image.save(output_path, quality=100, dpi=(300, 300))
    print(f"Image saved successfully at {output_path}")

# Example usage
# image_path_license = '/content/License-3.jpeg'
# image_path_passport = '/content/passport-2.jpg'
# correct_image_orientation(image_path_license)
# correct_image_orientation(image_path_passport)
