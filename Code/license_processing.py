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


class Address(BaseModel):
    street: str = Field(..., description="Street address")
    city: str = Field(..., description="City")
    state: str = Field(..., description="State (two-letter code)")
    zip_code: str = Field(..., description="ZIP code (5 or 9 digits)")

class LicenseData(BaseModel):
    full_name: str = Field(..., description="Full name of the license holder (Last, First Middle)")
    date_of_birth: str = Field(..., description="Date of birth (MM/DD/YYYY)")
    license_number: str = Field(..., description="License number (alphanumeric)")
    address: Address = Field(..., description="Address of the license holder")
    sex: str = Field(..., description="Sex/Gender (M, F, or X)")
    height: Optional[str] = Field(None, description="Height (e.g., 5'-11\")")
    weight: Optional[str] = Field(None, description="Weight (e.g., 180 lbs)")
    eye_color: Optional[str] = Field(None, description="Eye color (e.g., BRN, BLU)")
    hair_color: Optional[str] = Field(None, description="Hair color (e.g., BRN, BLK)")
    issuance_date: Optional[str] = Field(None, description="Date of issuance (MM/DD/YYYY)")
    expiration_date: str = Field(..., description="Expiration date (MM/DD/YYYY)")
    class_type: Optional[str] = Field(None, description="License class type (e.g., C, A, B)")
    # restrictions: Optional[str] = Field(None, description="License restrictions (if any)")
    # endorsements: Optional[str] = Field(None, description="License endorsements (if any)")

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
    prompt = f"""
    Analyze this driver's license image and extract the following information:
    1. Full name (Format: LAST NAME, First Name Middle Name)
    2. Date of birth (MM/DD/YYYY)
    3. License number (alphanumeric, including any leading letters like 'I' or 'DL')
    4. Complete address (street, city, state, ZIP code)
    5. Sex/Gender
    6. Height
    7. Weight
    8. Eye color
    9. Hair color
    10. Issuance date (MM/DD/YYYY) - This is typically present, make sure to extract if visible
    11. Expiration date (MM/DD/YYYY)
    12. License class type

    Provide the extracted information in a JSON format strictly adhering to the following schema:
    {LicenseData.schema_json(indent=2)}

    Important:
    - Extract only the information visible in the image.
    - Do not invent or assume any information not present.
    - If a field is not visible or not applicable, use null for optional fields.
    - Ensure all dates are in MM/DD/YYYY format.
    - Double-check the accuracy of all extracted information.
    - Pay special attention to the license number format, including any leading letters.
    - Make sure to extract the issuance date if it's visible on the license.
    """
    
    # Rest of the function remains the same
    payload = {
        "model": "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
        "max_tokens": 16384,
        "temperature": 0.2,
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
    prompt = """
    Extract and list all text visible in this image, line by line. Include everything you can see, such as:
    - All text on the front of the license
    - Any numbers, codes, or identifiers
    - Text in different fonts or sizes
    - Any watermarks or security features you can discern
    - Text orientation and placement
    
    Transcribe any handwritten or printed text you see in this image. Maintain the original formatting as much as possible.
    
    Identify and extract all text from this document image. Separate different sections or paragraphs clearly.
    
    Read and transcribe the text in this image. Also, describe the font style and text placement.
    
    Do not interpret or structure the information, just provide a raw, detailed transcription of all visible text and elements.
    """
    
    payload = {
        "model": "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
        "max_tokens": 16384,
        "temperature": 0.1,
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
    You are an expert in US driver's license validation. Your task is to validate and correct the information extracted from a driver's license image. Use the following step-by-step approach:

    1. Analyze the extracted JSON:
    {json.dumps(extracted_json, indent=2)}

    2. Compare it with the raw text extraction:
    {raw_text}

    3. For each field in the JSON:
       a. Check if it matches the information in the raw text.
       b. Verify if the format is correct (e.g., dates in MM/DD/YYYY format).
       c. Ensure the information is plausible for a US driver's license.

    4. Pay special attention to the following:
       a. Full Name: Ensure it follows the format "LAST NAME, First Name Middle Name". 
          The last name should be in all capital letters, followed by a comma, then the first name (and middle name if present).
       b. License Number: Ensure it includes any leading letters (e.g., 'I', 'DL', etc.) that are part of the full license number.
       c. Issuance Date: This field is often present on licenses. If it's visible in the raw text but missing in the JSON, make sure to include it.

    5. When validating the license number and dates:
       a. Check for any prefixes or formatting specific to the state (e.g., 'DL' or 'I' before numbers).
       b. Ensure all dates are in the correct MM/DD/YYYY format.
       c. Verify that the issuance date is present and precedes the expiration date.

    6. If you find any discrepancies or errors:
       a. Identify the correct information from the raw text.
       b. Explain your reasoning for making the correction.

    7. If any required fields are missing, attempt to fill them using the raw text.

    8. Reflect on your changes:
       a. Are all corrections justified by the available information?
       b. Have you maintained consistency across all fields?
       c. Are there any fields you're uncertain about? If so, explain why.

    9. Provide the final, validated JSON that strictly adheres to this schema:
    {LicenseData.schema_json(indent=2)}

    Remember:
    - Only include information that can be verified from the provided data.
    - If a field is truly missing or cannot be determined, use null for optional fields.
    - Explain any significant changes or decisions you make in the validation process.
    - Ensure the license number is complete and accurate, including any leading characters.
    - Make sure the issuance date is included if it's visible in the raw text.
    """

    # Rest of the function remains the same

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
