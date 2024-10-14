# passport_processing.py
import json
import base64
from PIL import Image
from io import BytesIO
import requests
from pprint import pprint
import os
import openai
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Access the API key
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY not found in environment variables. Please set it in your .env file or in your environment.")

# Set up the Fireworks client
client = openai.Client(api_key=API_KEY, base_url="https://api.fireworks.ai/inference/v1")

# Use API_KEY in your requests
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}


class MRZ(BaseModel):
    line1: str = Field(..., description="First line of MRZ (44 characters)")
    line2: str = Field(..., description="Second line of MRZ (44 characters)")

class PassportData(BaseModel):
    full_name: str = Field(..., description="Full name of the passport holder")
    date_of_birth: str = Field(..., description="Date of birth (DD MMM YYYY)")
    passport_number: str = Field(..., description="Passport number")
    nationality: str = Field(..., description="Nationality")
    place_of_birth: Optional[str] = Field(None, description="Place of birth")
    issuance_date: str = Field(..., description="Date of issuance (DD MMM YYYY)")
    expiration_date: str = Field(..., description="Expiration date (DD MMM YYYY)")
    sex: str = Field(..., description="Sex (M or F)")
    authority: str = Field(..., description="Issuing authority")
    mrz: MRZ = Field(..., description="Machine Readable Zone data")

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
    prompt = """
    Extract the following fields from the passport and provide them in a structured JSON format:
    - Full Name
    - Date of Birth (in format: DD MMM YYYY)
    - Passport Number
    - Nationality
    - Place of Birth (if visible)
    - Issuance Date (in format: DD MMM YYYY)
    - Expiration Date (in format: DD MMM YYYY)
    - Sex (M or F)
    - Authority (issuing authority)
    - MRZ (Machine Readable Zone)

    For the MRZ, provide the exact two lines of 44 characters each. Do not interpret the MRZ, just provide the raw data.

    Ensure that the JSON output is structured correctly and the fields are properly filled. 
    If a field is not visible or not applicable, use null. 
    Only provide data that can be verified from the image.
    """
    payload = {
        "model": "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
        "max_tokens": 16384,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
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
    print("Structured JSON extraction from LLaMA 11B including MRZ:")
    pprint(response.json())
    return response.json()

def extract_raw_text_from_llama11b(image_base64):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    prompt = "Extract all the information from this passport image, including the MRZ."
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
    print("Raw text extraction from LLaMA 11B including MRZ:")
    pprint(response.json())
    return response.json()

def validate_fields_with_llama405b(extracted_json, raw_text):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    validation_prompt = f"""
    You are an expert in passport validation. Your task is to validate and correct the extracted information from a passport image.
    
    Given the extracted JSON and raw text from the image, please:
    1. Verify the accuracy of each field.
    2. Correct any errors or inconsistencies.
    3. Fill in any missing information that can be inferred from the raw text.
    4. Ensure all dates are in the format DD MMM YYYY.
    5. Make sure the MRZ data consists of two lines of exactly 44 characters each.
    
    Extracted JSON: {json.dumps(extracted_json)}
    
    Raw Text from Image: {raw_text}
    
    Please provide the extracted and validated data in a JSON format. If a field is not available or not applicable, use null for optional fields or a placeholder for required fields.
    """
    payload = {
        "model": "accounts/fireworks/models/llama-v3p1-405b-instruct",
        "max_tokens": 16384,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
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
    print("Validation and correction response from LLaMA 405B:")
    validated_data = response.json()
    pprint(validated_data)
    return validated_data


def process_passport(image_path):
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
    validated_data = validate_fields_with_llama405b(extracted_json, raw_text)
    buffer[-1]["raw_output"] = validated_data

    # Step 5: Final output and validation
    buffer.append({
        "description": "Step 5: Final validated output",
        "raw_output": validated_data
    })

    # Validate against Pydantic model
    try:
        content = json.loads(validated_data['choices'][0]['message']['content'])
        # Ensure MRZ is properly formatted
        if 'mrz' in content and isinstance(content['mrz'], str):
            mrz_lines = content['mrz'].split('\n')
            if len(mrz_lines) == 2 and all(len(line) == 44 for line in mrz_lines):
                content['mrz'] = {'line1': mrz_lines[0], 'line2': mrz_lines[1]}
            else:
                raise ValueError("MRZ format is incorrect")
        final_data = PassportData(**content)
        return final_data.dict(), buffer
    except Exception as e:
        print(f"Error in validating result: {str(e)}")
        return content, buffer

