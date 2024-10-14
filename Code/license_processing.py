# license_processing.py
import json
import base64
from PIL import Image
from io import BytesIO
import requests
from pprint import pprint

import os
import openai
os.environ["FIREWORKS_API_KEY"] = "fw_3ZYfxHwhiBcN7MvMuZyemtjq"
API_KEY = os.environ.get("FIREWORKS_API_KEY")
client = openai.Client(api_key="fw_3ZZMiTWAZFcP2JQ6AjSyX3zz", base_url="https://api.fireworks.ai/inference/v1")


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
        "- Full Name\n"
        "- Date of Birth\n"
        "- License Number\n"
        "- Address (Street, City, State, Zip Code)\n"
        "- Sex\n"
        "- Height\n"
        "- Weight\n"
        "- Eye Color\n"
        "- Issuance Date\n"
        "- Expiration Date\n"
        "Ensure that the JSON output is structured correctly and the fields are properly filled. "
        "Do not hallucinate information. Only provide data that can be verified from the image."
    )
    payload = {
        "model": "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
        "max_tokens": 16384,
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

    validation_prompt = """
    You are an expert in US driver's license validation. Your task is to validate and extract information from a driver's license image, accommodating variations across different states.
    
    Given the extracted JSON and raw text from the image, please:
    1. Extract and validate the core information typically found on US driver's licenses.
    2. Include any additional fields that are present and relevant.
    3. Ensure accuracy and consistency of the extracted information.
    
    Focus on these common fields, but don't limit yourself to only these if more information is available:
    - Full Name
    - Date of Birth
    - License Number
    - Address (as detailed as possible)
    - Sex/Gender
    - Height
    - Eye Color
    - Issue Date
    - Expiration Date
    
    Extracted JSON: {extracted_json}
    
    Raw Text from Image: {raw_text}
    
    Please provide the extracted and validated data in a JSON format. Include any additional fields you find relevant or unique to this license. If a field is not available or not applicable, omit it from the JSON rather than including it with a null value.
    """

    payload = {
        "model": "accounts/fireworks/models/llama-v3p1-405b-instruct",
        "max_tokens": 16384,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": validation_prompt.format(extracted_json=json.dumps(extracted_json), raw_text=raw_text)
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



def format_with_llama45b(extracted_json, raw_text):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    formatting_prompt = (
        "Please take the following extracted fields and raw text from a driver's license "
        "and format it into a properly structured JSON format. Ensure that the output follows "
        "this schema:\n"
        "{\n"
        '"Full Name": "",\n'
        '"Date of Birth": "",\n'
        '"License Number": "",\n'
        '"Address": {\n'
        '  "Street": "",\n'
        '  "City": "",\n'
        '  "State": "",\n'
        '  "Zip Code": ""\n'
        '},\n'
        '"Sex": "",\n'
        '"Height": "",\n'
        '"Weight": "",\n'
        '"Eye Color": "",\n'
        '"Issuance Date": "",\n'
        '"Expiration Date": ""\n'
        "}\n"
        "Please ensure all fields are filled accurately based on the extracted data.\n"
        f"Extracted JSON: {json.dumps(extracted_json)}\n"
        f"Raw Text: {raw_text}\n"
    )
    payload = {
        "model": "accounts/fireworks/models/llama-v3p1-405b-instruct",
        "max_tokens": 16384,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": formatting_prompt}
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
    print("Formatted JSON response from LLaMA 4.5B:")
    pprint(response.json())
    return response.json()


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

    # Step 5: Format final output
    buffer.append({
        "description": "Step 5: Formatting the final validated output as a structured JSON...",
        "raw_output": validated_fields
    })

    return validated_fields, buffer





# Example usage
# if __name__ == "__main__":
#     image_path = '/content/License 1.png'
#     result = process_license(image_path)
#     print("Final Validated JSON Output:", result)