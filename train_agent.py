import os
import pickle
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create uploads and output folders
UPLOAD_FOLDER = "uploads"
INDEX_FILE = "vector_index.pkl"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(file_path):
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        # OCR fallback for scanned or handwritten PDFs
        if not text.strip():
            images = convert_from_path(file_path)
            for img in images:
                text += pytesseract.image_to_string(img)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text

def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
    return text

def create_vector_index(all_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(all_text)
    vectorstore = FAISS.from_texts(chunks, embedding_model)
    return vectorstore

def train_from_files():
    files = os.listdir(UPLOAD_FOLDER)
    if not files:
        print("❌ No files found in 'uploads' folder. Please add PDFs or DOCXs.")
        return

    all_text = ""
    for file in files:
        path = os.path.join(UPLOAD_FOLDER, file)
        if file.endswith(".pdf"):
            all_text += extract_text_from_pdf(path)
        elif file.endswith(".docx"):
            all_text += extract_text_from_docx(path)
        else:
            print(f"Skipping unsupported file: {file}")

    print("✅ Files processed. Creating FAISS index...")
    vectorstore = create_vector_index(all_text)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump(vectorstore, f)

    print(f"✅ Index saved as {INDEX_FILE}")

if __name__ == "__main__":
    train_from_files()
