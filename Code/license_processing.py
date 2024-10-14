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
    full_name: str = Field(..., description="Full name of the license holder")
    date_of_birth: str = Field(..., description="Date of birth")
    license_number: str = Field(..., description="License number")
    address: Address = Field(..., description="Address of the license holder")
    sex: str = Field(..., description="Sex/Gender")
    height: Optional[str] = Field(None, description="Height")
    weight: Optional[str] = Field(None, description="Weight")
    eye_color: Optional[str] = Field(None, description="Eye color")
    issuance_date: Optional[str] = Field(None, description="Date of issuance")
    expiration_date: str = Field(..., description="Expiration date")

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
    prompt = (
        "Extract the following fields from the driver's license and provide them in a structured JSON format:\n"
        f"{LicenseData.schema_json(indent=2)}\n"
        "Ensure that the JSON output is structured correctly and the fields are properly filled. "
        "Do not hallucinate information. Only provide data that can be verified from the image."
    )
    payload = {
        "model": "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
        "max_tokens": 16384,
        "temperature": 0.1,
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
    print("Structured JSON extraction from LLaMA 11B:")
    pprint(response.json())
    return response.json()

def extract_raw_text_from_llama11b(image_base64):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    prompt = "Extract all the information from this image"
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
    print("Raw text extraction from LLaMA 11B:")
    pprint(response.json())
    return response.json()

def validate_fields_with_llama405b(extracted_json, raw_text):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"

    validation_prompt = f"""
    You are an expert in US driver's license validation. Your task is to validate and extract information from a driver's license image, accommodating variations across different states.
    
    Given the extracted JSON and raw text from the image, please:
    1. Extract and validate the core information typically found on US driver's licenses.
    2. Include any additional fields that are present and relevant.
    3. Ensure accuracy and consistency of the extracted information.
    
    Use the following schema for the output:
    {LicenseData.schema_json(indent=2)}
    
    Extracted JSON: {json.dumps(extracted_json)}
    
    Raw Text from Image: {raw_text}
    
    Please provide the extracted and validated data in a JSON format that strictly adheres to the given schema. If a field is not available or not applicable, use null for optional fields or a placeholder for required fields.
    """

    payload = {
        "model": "accounts/fireworks/models/llama-v3p1-405b-instruct",
        "max_tokens": 16384,
        "temperature": 0.2,
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

    print("Validation and extraction response from LLaMA 405B:")
    validated_data = response.json()
    pprint(validated_data)

    # Extract the content from the response
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
