import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import os
from streamlit_cropper import st_cropper
from orientation import correct_image_orientation, get_orientation_from_llama
from license_processing import process_license
from passport_processing import process_passport
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

# Use API_KEY in your requests
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}



# Try to import LicenseData, but don't fail if it's not available
try:
    from license_processing import LicenseData
    USE_LICENSE_DATA = True
except ImportError:
    USE_LICENSE_DATA = False
    st.warning("LicenseData model not available. Validation will be skipped.")

def encode_image_base64(img):
    if img.mode in ('RGBA', 'LA'):
        img = img.convert('RGB')
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def main():
    st.title("Document Processing App")

    # Document type selection
    doc_type = st.radio("Select document type:", ("Passport", "Driver's License"))

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Orientation check and correction
        if st.button("Check and Correct Orientation"):
            with st.spinner("Checking orientation..."):
                image_base64 = encode_image_base64(image)
                orientation = get_orientation_from_llama(image_base64)
                
                if orientation == 0:
                    st.write("Image orientation is correct.")
                else:
                    st.write(f"Correcting orientation by {orientation} degrees...")
                    corrected_image = image.rotate(-orientation, expand=True)
                    st.image(corrected_image, caption="Corrected Image", use_column_width=True)
                    image = corrected_image

        # Manual orientation correction with fixed values (multiples of 90)
        st.write("If the orientation is still incorrect, you can manually adjust it:")
        manual_angle = st.select_slider(
            "Rotation angle", 
            options=[-90, 0, 90, 180, 270],
            value=0
        )
        if manual_angle != 0:
            manually_corrected_image = image.rotate(-manual_angle, expand=True)
            st.image(manually_corrected_image, caption="Manually Corrected Image", use_column_width=True)
            image = manually_corrected_image

        # Image cropping
        st.write("Select region of interest (click and drag on the image):")
        st.write("You can move the red box by dragging the four corners or the sides to adjust the area you want to select.")
        cropped_img = st_cropper(image, realtime_update=True, box_color='red', aspect_ratio=None)

        if cropped_img is not None:
            st.image(cropped_img, caption="Cropped Image", use_column_width=True)
            image = cropped_img  # Use the cropped image for further processing

        # Document processing
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                # Save the image temporarily
                temp_image_path = "temp_image.jpg"
                
                # Convert RGBA to RGB if necessary
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                image.save(temp_image_path)
                
                try:
                    if doc_type == "Passport":
                        result, buffer = process_passport(temp_image_path)
                    else:  # Driver's License
                        result, buffer = process_license(temp_image_path)

                    # Display processing steps
                    with st.expander("View Processing Steps", expanded=True):
                        for step in buffer:
                            st.write(step['description'])

                    # Display results
                    st.success("Document processed successfully!")
                    st.subheader("Extracted Information")
                    if doc_type == "Driver's License" and USE_LICENSE_DATA:
                        # Validate the result against the LicenseData model
                        try:
                            validated_result = LicenseData(**result)
                            st.json(validated_result.dict())
                        except Exception as e:
                            st.error(f"Error in validating result: {str(e)}")
                            st.json(result)
                    else:
                        st.json(result)

                    # Display raw output in sidebar
                    st.sidebar.subheader("Raw Output")
                    for i, step_output in enumerate(buffer):
                        with st.sidebar.expander(f"Step {i+1} Raw Output"):
                            st.sidebar.json(step_output['raw_output'])

                except Exception as e:
                    st.error(f"Error during document processing: {str(e)}")
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)

    st.sidebar.header("About")
    st.sidebar.info("This app processes passport and driver's license documents using AI.")

if __name__ == "__main__":
    main()
