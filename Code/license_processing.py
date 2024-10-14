# license_processing.py
import json
import base64
from PIL import Image
from io import BytesIO
import requests
from pprint import pprint
from pydantic import BaseModel, Field
from typing import Optional
import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Access the API key
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY not found in environment variables. Please set it in your .env file or in your environment.")

# Set up the Fireworks client
client = openai.Client(api_key=API_KEY, base_url="https://api.fireworks.ai/inference/v1")



# Rest of your license_processing.py code...
class Address(BaseModel):
    street: str = Field(..., description="Street address")
    city: str = Field(..., description="City")
    state: str = Field(..., description="State")
    zip_code: str = Field(..., description="ZIP code")

class LicenseData(BaseModel):
    full_name: Optional[str] = Field(None, description="Full name of the license holder")
    date_of_birth: Optional[str] = Field(None, description="Date of birth (DD/MM/YYYY format or similar)")
    license_number: Optional[str] = Field(None, description="Driver's license number")
    address: Optional[Address] = Field(None, description="Address of the license holder")
    sex: Optional[str] = Field(None, description="Sex/Gender of the license holder")
    height: Optional[str] = Field(None, description="Height of the license holder")
    weight: Optional[str] = Field(None, description="Weight of the license holder (if available)")
    eye_color: Optional[str] = Field(None, description="Eye color")
    issuance_date: Optional[str] = Field(None, description="Date of issuance (ISS, Issue Date, etc.)")
    expiration_date: Optional[str] = Field(None, description="Date of expiration (this may vary by format)")
    license_class: Optional[str] = Field(None, description="License class (A, B, C, etc.)")
    endorsements: Optional[str] = Field(None, description="Endorsements (special privileges for the driver)")
    restrictions: Optional[str] = Field(None, description="Restrictions (any limitations or conditions on the license)")



def encode_image_base64(img):
    if img.mode in ('RGBA', 'LA'):
        img = img.convert('RGB')
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def encode_image_direct(image_path):
    with Image.open(image_path) as img:
        return encode_image_base64(img)

def extract_json_from_llama11b(image_base64):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    
    # Generalized prompt for chain-of-thought reasoning to extract all relevant fields
    prompt = (
        "You are tasked with extracting fields from a driver's license. The license may have variations in naming, "
        "so identify and extract any fields that correspond to the following JSON structure:\n"
        f"{LicenseData.schema_json(indent=2)}\n"
        "Fields like 'issuance date', 'license class', or 'endorsements' may be labeled differently (e.g., 'ISS' for issuance date). "
        "Ensure that the output is accurate and that any optional fields missing from the license are returned as null. "
        "If certain fields are ambiguous, use your best reasoning to resolve them."
    )
    
    payload = {
        "model": "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
        "max_tokens": 16384,
        "temperature": 0.1,  # Lower temperature for more deterministic results
        "response_format": {"type": "json_object", "schema": LicenseData.schema_json()},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    pprint(response.json())
    return response.json()


def extract_raw_text_from_llama11b(image_base64):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    
    prompt = "Extract all the information (raw text) from this image of a driver's license, regardless of field formatting or position."

    payload = {
        "model": "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
        "max_tokens": 16384,
        "response_format": {"type": "text"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    pprint(response.json())
    return response.json()


def validate_fields_with_llama405b(extracted_json, raw_text):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"

    validation_prompt = f"""
    Extracted JSON: {json.dumps(extracted_json)}
    Raw Text from Image: {raw_text}

    You are an expert in US driver's license validation. Your task is to validate the extracted data and handle variations in field naming across different states.

    1. Validate each field based on the context of the raw text.
    2. If a field is missing or ambiguous, reason through the most appropriate match from the raw text and context.
    3. Ensure that any fields like 'issuance date' or 'license class' that may appear under alternative names are extracted and validated correctly.
    4. Return the final validated and corrected JSON output, adhering to the following schema:
    {LicenseData.schema_json(indent=2)}

    
    """

    payload = {
        "model": "accounts/fireworks/models/llama-v3p1-405b-instruct",
        "max_tokens": 16384,
        "temperature": 0.5,
        "response_format": {"type": "json_object", "schema": LicenseData.schema_json()},
        "messages": [
            {
                "role": "user",
                "content": validation_prompt
            }
        ]
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    validated_data = response.json()
    validated_json = json.loads(validated_data['choices'][0]['message']['content'])

    return validated_json



def process_license(image_path):
    buffer = []

    # Step 1: Encode the image
    buffer.append({
        "description": "Step 1: Encoding the image...",
        "raw_output": {"status": "Image encoded successfully"}
    })
    image_base64 = encode_image_direct(image_path)

    # Step 2: Extract structured JSON
    buffer.append({
        "description": "Step 2: Extracting structured information using LLaMA Vision 11B model...",
        "raw_output": {}
    })
    extracted_json = extract_json_from_llama11b(image_base64)
    buffer[-1]["raw_output"] = extracted_json

    # Step 3: Extract raw text
    buffer.append({
        "description": "Step 3: Extracting raw text from the image using LLaMA Vision 11B model...",
        "raw_output": {}
    })
    raw_text = extract_raw_text_from_llama11b(image_base64)
    buffer[-1]["raw_output"] = raw_text

    # Step 4: Validate fields
    buffer.append({
        "description": "Step 4: Validating and correcting extracted information using LLaMA 405B model...",
        "raw_output": {}
    })
    validated_fields = validate_fields_with_llama405b(extracted_json, raw_text)
    buffer[-1]["raw_output"] = validated_fields

    # Step 5: Final output
    buffer.append({
        "description": "Step 5: Final validated output",
        "raw_output": validated_fields
    })

    return validated_fields, buffer

# # Example usage
# if __name__ == "__main__":
#     image_path = '/path/to/your/license/image.jpg'
#     result, process_buffer = process_license(image_path)
#     print("Final Validated JSON Output:")
#     pprint(result)
    
#     print("\nProcessing Steps:")
#     for step in process_buffer:
#         print(f"\n{step['description']}")
#         pprint(step['raw_output'])
