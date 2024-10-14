# passport_processing.py
import json
import base64
from PIL import Image
from io import BytesIO
import requests
from pprint import pprint
from pydantic import BaseModel, Field
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Access the API key
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY not found in environment variables. Please set it in your .env file or in your environment.")

class MRZ(BaseModel):
    line1: str = Field(..., description="First line of MRZ (44 characters)")
    line2: str = Field(..., description="Second line of MRZ (44 characters)")

class PassportData(BaseModel):
    full_name: Optional[str] = Field(..., description="Full name of the passport holder")
    date_of_birth: Optional[str] = Field(..., description="Date of birth (DD MMM YYYY)")
    passport_number: Optional[str] = Field(..., description="Passport number")
    nationality: Optional[str] = Field(..., description="Nationality")
    place_of_birth: Optional[str] = Field(None, description="Place of birth")
    issuance_date: Optional[str] = Field(None, description="Date of issuance (DD MMM YYYY)")
    expiration_date: Optional[str] = Field(..., description="Expiration date (DD MMM YYYY)")
    sex: Optional[str] = Field(..., description="Sex (M or F)")
    authority: Optional[str] = Field(None, description="Issuing authority")
    mrz: Optional[dict] = Field(None, description="Machine Readable Zone data")

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
    Analyze this passport image and extract the following information:
    - Full name of the passport holder
    - Date of birth (in format: DD MMM YYYY)
    - Passport number
    - Nationality
    - Place of birth (if visible)
    - Issuance date (in format: DD MMM YYYY)
    - Expiration date (in format: DD MMM YYYY)
    - Sex (M or F)
    - Authority (issuing authority)
    - MRZ (Machine Readable Zone)

    For the MRZ:
    1. Provide the exact two lines of 44 characters each.
    2. Do not interpret the MRZ, just provide the raw text.
    3. Ensure each line is exactly 44 characters long.
    4. The MRZ should only contain uppercase letters, numbers, and '<' symbols.

    Provide the extracted information in a JSON format strictly adhering to the following schema:
    {PassportData.schema_json(indent=2)}

    Example MRZ format:
    "mrz": {{
        "line1": "P<USASMITH<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
        "line2": "1234567890USA6802034M1509048<<<<<<<<<<<<<<02"
    }}

    Important:
    - Extract only the information visible in the image.
    - Do not invent or assume any information not present.
    - If a field is not visible or not applicable, use null for optional fields.
    - Ensure all dates are in DD MMM YYYY format.
    - Double-check the accuracy of all extracted information.
    """
    
    payload = {
        "model": "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
        "max_tokens": 16384,
        "temperature": 0.1,
        "response_format": {"type": "json_object", "schema": PassportData.schema_json()},
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
    return response.json()


    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    prompt = """
    Extract and list all text visible in this passport image, line by line. Include everything you can see, such as:
    - All text on the passport page
    - Any numbers, codes, or identifiers
    - Text in different fonts or sizes
    - Any watermarks or security features you can discern
    - Text orientation and placement
    
    Pay special attention to the Machine Readable Zone (MRZ) at the bottom of the passport page. 
    Transcribe it exactly as it appears, maintaining the original formatting.
    
    Example MRZ format:
    P<USASMITH<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<
    1234567890USA6802034M1509048<<<<<<<<<<<<<<02
    
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
    return response.json()

def extract_raw_text_from_llama11b(image_base64):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    prompt = """
    Extract and list all text visible in this passport image, line by line. Include everything you can see, such as:
    - All text on the passport page
    - Any numbers, codes, or identifiers
    - Text in different fonts or sizes
    - Any watermarks or security features you can discern
    - Text orientation and placement
    
    Provide the extracted text as plain text without any formatting or markdown syntax.
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
    return response.json()


def validate_fields_with_llama405b(extracted_json, raw_text):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    validation_prompt = f"""
    You are an expert in passport validation. Your task is to validate and correct the information extracted from a passport image. Use the following step-by-step approach:

    1. Analyze the extracted JSON:
    {json.dumps(extracted_json, indent=2)}

    2. Compare it with the raw text extraction:
    {raw_text}

    3. For each field in the JSON:
       a. Check if it matches the information in the raw text.
       b. Verify if the format is correct (e.g., dates in DD MMM YYYY format).
       c. Ensure the information is plausible for a passport.

    4. Pay special attention to the MRZ (Machine Readable Zone):
       - The MRZ for a passport should consist of two lines, each 44 characters long.
       - It should only contain uppercase letters, numbers, and '<' symbols.
       - The first line typically starts with 'P<' followed by the issuing country code.
       - The second line contains the passport number, date of birth, and expiration date in a specific format.

    Example of a valid MRZ:
    P<USASMITH<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<
    1234567890USA6802034M1509048<<<<<<<<<<<<<<02

    5. If the MRZ is missing or incomplete, attempt to construct it based on the available information.

    6. If you find any discrepancies or errors in any field:
       a. Identify the correct information from the raw text or your analysis.
       b. Explain your reasoning for making the correction.
       c. Provide the corrected information.

    7. Reflect on your entire validation process:
       a. Are all corrections and validations justified by the available information?
       b. Have you maintained consistency across all fields?
       c. Are there any fields or aspects you're uncertain about? If so, explain why.

    8. Provide the final, validated JSON that adheres to the passport data structure, including any corrections or additional information you've determined.

    Remember:
    - Only include information that can be verified or reasonably inferred from the provided data.
    - If a field is truly missing or cannot be determined, use null for optional fields.
    - Explain any significant changes or decisions you make in the validation process.
    - Ensure the MRZ is complete, accurately transcribed, and consistent with other passport data.
    """

    payload = {
        "model": "accounts/fireworks/models/llama-v3p1-405b-instruct",
        "max_tokens": 16384,
        "temperature": 0.2,
        "response_format": {"type": "json_object", "schema": PassportData.schema_json()},
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
    return json.loads(validated_data['choices'][0]['message']['content'])


def process_passport(image_path):
    buffer = []

    try:
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
        extracted_content = json.loads(extracted_json['choices'][0]['message']['content'])
        buffer[-1]["raw_output"] = extracted_content

        # Step 3: Extract raw text
        buffer.append({
            "description": "Step 3: Extracting raw text from the image...",
            "raw_output": {}
        })
        raw_text_response = extract_raw_text_from_llama11b(image_base64)
        raw_content = raw_text_response['choices'][0]['message']['content']
        buffer[-1]["raw_output"] = {"raw_text": raw_content}

        # Step 4: Validate and correct fields
        buffer.append({
            "description": "Step 4: Validating and correcting extracted information using LLaMA 405B model...",
            "raw_output": {}
        })
        validated_data = validate_fields_with_llama405b(extracted_content, raw_content)
        buffer[-1]["raw_output"] = validated_data

        # Step 5: Create PassportData object
        passport_data = PassportData(**validated_data)
        
        return passport_data.dict(), buffer

    except Exception as e:
        print(f"Error in processing passport: {str(e)}")
        buffer.append({
            "description": "Error in processing",
            "raw_output": {"error": str(e)}
        })
        return None, buffer
    buffer = []

    try:
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
        extracted_content = json.loads(extracted_json['choices'][0]['message']['content'])
        buffer[-1]["raw_output"] = extracted_content

        # Step 3: Extract raw text
        buffer.append({
            "description": "Step 3: Extracting raw text from the image...",
            "raw_output": {}
        })
        raw_text = extract_raw_text_from_llama11b(image_base64)
        raw_content = raw_text['choices'][0]['message']['content']
        buffer[-1]["raw_output"] = raw_content

        # Step 4: Validate and correct fields
        buffer.append({
            "description": "Step 4: Validating and correcting extracted information using LLaMA 405B model...",
            "raw_output": {}
        })
        validated_data = validate_fields_with_llama405b(extracted_content, raw_content)
        buffer[-1]["raw_output"] = validated_data

        # Step 5: Create PassportData object
        passport_data = PassportData(**validated_data)
        
        return passport_data.dict(), buffer

    except Exception as e:
        print(f"Error in processing passport: {str(e)}")
        buffer.append({
            "description": "Error in processing",
            "raw_output": {"error": str(e)}
        })
        return None, buffer
    buffer = []

    try:
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
            "description": "Step 3: Extracting raw text from the image...",
            "raw_output": {}
        })
        raw_text = extract_raw_text_from_llama11b(image_base64)
        buffer[-1]["raw_output"] = raw_text

        # Step 4: Validate and correct fields
        buffer.append({
            "description": "Step 4: Validating and correcting extracted information using LLaMA 405B model...",
            "raw_output": {}
        })
        validated_data = validate_fields_with_llama405b(extracted_json, raw_text)
        buffer[-1]["raw_output"] = validated_data

        # Step 5: Create PassportData object
        passport_data = PassportData(**validated_data)
        
        return passport_data.dict(), buffer

    except Exception as e:
        print(f"Error in processing passport: {str(e)}")
        buffer.append({
            "description": "Error in processing",
            "raw_output": {"error": str(e)}
        })
        return None, buffer

# # Example usage
# if __name__ == "__main__":
#     image_path = '/path/to/your/passport/image.jpg'
#     result, process_buffer = process_passport(image_path)
#     print("Final Validated JSON Output:")
#     pprint(result)
    
#     print("\nProcessing Steps:")
#     for step in process_buffer:
#         print(f"\n{step['description']}")
#         pprint(step['raw_output'])
