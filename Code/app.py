# app.py
import streamlit as st
from PIL import Image
import io
import base64
import os
from io import BytesIO
from orientation import correct_image_orientation, get_orientation_from_llama
from license_processing import process_license
from passport_processing import process_passport

def encode_image_base64(img):
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

        # Manual orientation correction
        st.write("If the orientation is still incorrect, you can manually adjust it:")
        manual_angle = st.slider("Rotation angle", -180, 180, 0)
        if manual_angle != 0:
            manually_corrected_image = image.rotate(-manual_angle, expand=True)
            st.image(manually_corrected_image, caption="Manually Corrected Image", use_column_width=True)
            image = manually_corrected_image

        # Document processing
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                # Save the image temporarily
                temp_image_path = "temp_image.jpg"
                image.save(temp_image_path)
                
                try:
                    if doc_type == "Passport":
                        result, buffer = process_passport(temp_image_path)
                    else:  # Driver's License
                        result, buffer = process_license(temp_image_path)

                    # Display processing steps in main area
                    with st.expander("View Processing Steps", expanded=True):
                        for step in buffer:
                            st.write(step['description'])

                    # Display results
                    st.success("Document processed successfully!")
                    st.subheader("Extracted Information")
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