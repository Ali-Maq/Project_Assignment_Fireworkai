# AI-Driven KYC Document Processing System
![System Architecture](images/diagram-export-10-13-2024-7_44_03-PM.png)
This repository contains my implementation of an AI-driven system for processing Know Your Customer (KYC) documents. This project was part of an interview assignment provided by Fireworks AI. It is designed to automate the identity verification process for various document types, such as passports and driver's licenses, leveraging advanced AI techniques for information extraction, validation, and structuring. Below, you will find a detailed description of the components, methodologies, and system architecture employed in this project.

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Pipeline Components](#pipeline-components)
4. [Challenges and Solutions](#challenges-and-solutions)
5. [Future Directions](#future-directions)
6. [How to Run](#how-to-run)

## Overview
The goal of this project is to automate and enhance the KYC process using AI models, focusing on accuracy, efficiency, and adaptability across different document types. The system follows a multi-model pipeline approach to achieve high accuracy in extracting, validating, and structuring information from documents. Key features include:

- **Automatic image orientation correction** for improved model performance.
- **Multi-model pipeline** combining different large language models to enhance accuracy.
- **Manual correction fallback** to handle document cases where automated methods may fail.

## System Architecture

Below is a textual representation of the system architecture that helps illustrate the data flow:

```
            +----------------------------+
            |       Start Process        |
            +------------+---------------+
                         |
                         v
            +----------------------------+
            |   1. Image Upload &        |
            |      Preprocessing         |
            +------------+---------------+
                         |
                         v
            +----------------------------+
            |   2. Document Type         |
            |      Selection             |
            +------------+---------------+
                         |
                         v
            +----------------------------+
            |   3. Image Orientation     |
            |      Correction            |
            +------------+---------------+
                         |
                         v
            +----------------------------+
            |   4. Data Extraction       |
            |      & Structuring         |
            +------------+---------------+
                         |
                         v
            +----------------------------+
            |   5. Data Validation &     |
            |      Final Output          |
            +----------------------------+
```

The AI-driven KYC document processing system architecture consists of the following main steps:

1. **Image Upload and Preprocessing**: Handles user-uploaded images, ensuring all image formats (JPEG, PNG, etc.) are supported.
2. **Document Type Selection**: Allows users to choose between processing a passport or a driver's license.
3. **Image Orientation Correction**: Uses a language model to determine the document orientation for improved accuracy.
4. **Data Extraction and Structuring**: Involves extracting raw text from the document, generating structured JSON, and validating the extracted data.
5. **Validation and Final Output**: Uses a larger language model for data validation and generates a structured output in JSON format.

For better understanding, please refer to the diagram below, which visually represents the system architecture and flow of the KYC document processing system.


![System Architecture](images/diagram-export-10-13-2024-7_44_03-PM.png)


## Pipeline Components

### 1. Initial Image Processing

```
            +----------------------------+
            |  Image Upload &            |
            |  Preprocessing             |
            +----------------------------+
```
- **Upload and Preprocess**: The system starts with uploading an image, converting it to the appropriate format, and checking for correct orientation.
- **Orientation Detection and Correction**: The `orientation.py` script iteratively sends images to a LLaMA model to determine the correct orientation and rotates accordingly. A fallback manual correction feature is also available.

### 2. Raw Data Extraction

```
            +----------------------------+
            |  Passport & License        |
            |  Processing                |
            +----------------------------+
```
- **Passport and License Processing**: The `passport_processing.py` and `license_processing.py` scripts handle data extraction from specific document types. The LLaMA 3.2 11B vision model is utilized to extract raw text data.

### 3. Structured JSON Generation

```
            +----------------------------+
            |  Generate Structured       |
            |  JSON                      |
            +----------------------------+
```
- After extracting raw text, the system employs a language model to convert this data into a structured JSON format.

### 4. Data Validation and Refinement

```
            +----------------------------+
            |  Validate Data using       |
            |  LLaMA 405B Model          |
            +----------------------------+
```
- **Validation Using LLaMA 405B Model**: To ensure data consistency and accuracy, the extracted data is validated by a LLaMA 405B model. This stage involves reconciling discrepancies and refining the structured JSON output.

### 5. User Interface and Manual Adjustment

```
            +----------------------------+
            |  User Interface for        |
            |  Upload & Manual Adjust.   |
            +----------------------------+
```
- **Streamlit App**: The `app.py` script is used to implement the user interface where users can upload documents, manually adjust the orientation if needed, and view the results.
- **Manual Orientation Adjustment**: Users can manually adjust the document's rotation using a slider feature if the automatic orientation detection doesn't yield the correct result.

## Challenges and Solutions

### 1. Image Orientation
- **Challenge**: Incorrect orientation led to degraded model performance.
- **Solution**: A custom orientation detection system was developed, which uses a LLaMA model to determine the correct orientation. Additionally, a manual correction option is provided for further adjustment.

### 2. Data Consistency
- **Challenge**: There were inconsistencies between raw data extraction and structured JSON generation.
- **Solution**: A data validation step using a LLaMA 405B model was implemented to ensure consistency and accuracy.

### 3. Single Model Limitations
- **Challenge**: Single-model approaches did not achieve the desired accuracy.
- **Solution**: Developed a multi-model pipeline that combined different models for better accuracy and robustness.

## Future Directions

1. **Model Fine-Tuning**: Future plans include fine-tuning the LLaMA vision model on a diverse set of KYC documents for improved performance.
2. **Expanded Document Support**: Expand the system to handle a wider variety of documents, such as national ID cards and utility bills.
3. **Real-Time Integration**: Implement API connections to external services for cross-referencing extracted data and real-time validation.
4. **Enhanced UI/UX**: Improve user feedback mechanisms, batch processing capabilities, and enhanced error handling.

## How to Run

### Deploying the Streamlit App
To deploy this Streamlit app, you can use **Streamlit Cloud** or any other cloud service that supports Python applications. Below are the steps to deploy using Streamlit Cloud:

1. **Create an Account**: Go to [Streamlit Cloud](https://streamlit.io/cloud) and create an account.
2. **Link Your GitHub Repository**: Connect your Streamlit Cloud account to your GitHub and select this repository for deployment.
3. **Deploy**: After linking, click on 'Deploy an app', select the repository, and specify the path to `app.py` as the entry point.
4. **Configure Settings**: Ensure that environment variables such as API keys for Fireworks AI are properly set in the deployment settings.
5. **Access the App**: Once deployed, you will get a URL to access the app online.

### Prerequisites
- Python 3.8 or later
- Dependencies as specified in `requirements.txt`
- API access to Fireworks AI

### Steps to Run
1. **Clone the Repository**:
   ```sh
   git clone https://github.com/Ali-Maq/Project_Assignment_Fireworkai
   cd kyc-document-processing
   ```
2. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the Streamlit App**:
   ```sh
   streamlit run app.py
   ```
4. **Upload a Document**: Once the app is running, upload a passport or driver's license for processing.
5. **View Results**: The app will display the extracted and validated data in JSON format.

Feel free to explore the code, provide feedback, and suggest improvements. This project demonstrates the capabilities of using large language models for automating document processing tasks in the KYC domain. I am eager to see how it can evolve further with more feedback and collaboration.
