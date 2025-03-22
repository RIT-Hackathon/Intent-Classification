from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import torch
from pymongo import MongoClient
from dotenv import load_dotenv
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pickle
import requests
import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import os

# Load environment variables
load_dotenv()

# Get values from .env
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

# Load Model & Tokenizer
model_path = "./roberta_intent_model"  # Update this with your actual model path
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(model_path)

# Load Label Encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Move model to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define FastAPI App
app = FastAPI()

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Request Body Schema


class ExtractRequest(BaseModel):
    user_id: str
    file_path: str | None = None
    file_url: HttpUrl | None = None


class PredictRequest(BaseModel):
    query: str


class SuggestRequest(BaseModel):
    user_id: str
    user_query: str


class MedicalQueryRequest(BaseModel):
    user_query: str

# Function to download file from URL


def download_file(url):
    local_filename = url.split("/")[-1]
    file_path = f"./downloads/{local_filename}"

    os.makedirs("./downloads", exist_ok=True)  # Ensure directory exists

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return file_path
    else:
        raise HTTPException(status_code=400, detail="Failed to download file.")

# Extract text from image (JPG, PNG)


def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Extract text from PDF


def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Extract text from TXT


def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as file:
        return file.read()

# Extract text dynamically based on file type


def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        return extract_text_from_image(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

# Analyze medical report using Gemini API


def analyze_medical_report(text):
    headers = {"Content-Type": "application/json"}
    prompt = f"""
    Extract key medical details from the document and return a **concise** summary.

    **Format:**
    "Test Results: [test_name (value unit), ...], Diagnosis: [...], Medications: [...], Doctor Notes: '', Suggestions: [...]"

    **Rules:**
    - No extra text, explanations, or formatting—return only a **valid** string.
    - Keep "Suggestions" short (e.g., "Monitor blood sugar", "Increase hydration").
    - If no data is available for a section, leave it blank.

    **Document Content:**
    {text}
    """

    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return "Error: Failed to generate summary."

# Predict intent using the model


def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding="max_length", max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return label_encoder.inverse_transform([predicted_label])[0]

# 1️⃣ Extract & Store Medical Data


@app.post("/extract")
def extract_and_store(request: ExtractRequest):
    file_path = request.file_path

    # If a URL is provided, download the file first
    if request.file_url:
        file_path = download_file(str(request.file_url))

    if not file_path:
        raise HTTPException(status_code=400, detail="No file provided.")

    text = extract_text_from_file(file_path)
    summary = analyze_medical_report(text)

    print(f"Analysis result: {summary}")  # Debugging

    medical_data = {"user_id": request.user_id, "summary": summary}
    collection.insert_one(medical_data)

    return {"message": "Data extracted and stored successfully.", "summary": summary}

# 2️⃣ Upload & Suggest Based on History


@app.post("/upload_and_suggest")
def upload_and_suggest(request: ExtractRequest):
    file_path = request.file_path

    # If a URL is provided, download the file first
    if request.file_url:
        file_path = download_file(str(request.file_url))

    if not file_path:
        raise HTTPException(status_code=400, detail="No file provided.")

    text = extract_text_from_file(file_path)
    new_summary = analyze_medical_report(text)

    if "Error" in new_summary:
        return {"error": new_summary}

    previous_records = list(collection.find(
        {"user_id": request.user_id}, {"_id": 0, "user_id": 0}))

    # Generate AI suggestions based on history
    headers = {"Content-Type": "application/json"}
    prompt = f"""
    Based on past medical summaries, provide new suggestions for next steps or any dietary suggestions.

    **Previous Records:**
    {previous_records}

    **New Summary:**
    {new_summary}

    **Format:**
    "Suggestions: [...]"

    Provide only **plain text**.
    """

    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        ai_response = response.json(
        )["candidates"][0]["content"]["parts"][0]["text"]
        new_summary += f"\n{ai_response}"
    else:
        ai_response = "Error: Failed to generate suggestions."

    collection.insert_one({"user_id": request.user_id, "summary": new_summary})

    return {"message": "Data extracted and suggestions generated.", "summary": new_summary}

# 3️⃣ Predict Intent


@app.post("/predict")
async def get_intent(request: PredictRequest):
    intent = predict_intent(request.query)
    return {"query": request.query, "intent": intent}

# 4️⃣ Fetch Previous Suggestions


@app.post("/suggest")
def suggest(request: SuggestRequest):
    previous_records = list(collection.find(
        {"user_id": request.user_id}, {"_id": 0, "user_id": 0}))

    if not previous_records:
        return {"message": "No medical records found for this user."}

    # AI-based personalized suggestion
    headers = {"Content-Type": "application/json"}
    prompt = f"""
    Based on past medical summaries, provide concise suggestions for: **"{request.user_query}"**.

    **Medical Records:**
    {previous_records}

    **Format:**
    "Suggestions: [...]"

    Provide only **plain text**.
    """

    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        return {"suggestions": response.json()["candidates"][0]["content"]["parts"][0]["text"]}
    else:
        return {"error": "Failed to generate suggestions."}


@app.post("/medical-query")
def medical_query(request: MedicalQueryRequest):
    # AI-based personalized suggestion
    print(request)
    headers = {"Content-Type": "application/json"}
    prompt = f"""
    {request.user_query} Provide a short and concise answer to this
    If necessary, include general advice, but **avoid providing diagnoses**.  
    """

    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        return {"suggestions": response.json()["candidates"][0]["content"]["parts"][0]["text"]}
    else:
        return {"error": "Failed to generate suggestions."}
