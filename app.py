import os
import pickle
from dotenv import load_dotenv
from flask import Flask, render_template, request
import google.generativeai as genai
from markdown2 import markdown
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# ✅ Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Global variables
INDEX_FILE = "vector_index.pkl"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Load FAISS embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Load index if available
vectorstore = None
if os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, "rb") as f:
        vectorstore = pickle.load(f)

# ✅ Gemini model
model = genai.GenerativeModel("models/gemini-2.5-flash")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Handles file upload and optional query processing.
    If a file is uploaded, retrains and saves FAISS index.
    If no file is uploaded, reuses the previous one in memory.
    """
    global vectorstore

    uploaded_files = request.files.getlist("files")
    query = request.form.get("query", "").strip()

    # ✅ Step 1: Only rebuild index when new files are uploaded
    if uploaded_files and uploaded_files[0].filename != "":
        for file in uploaded_files:
            save_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(save_path)

        # ✅ Run training script to build FAISS index
        os.system("python train_agent.py")

        # ✅ Load updated FAISS index into memory
        with open(INDEX_FILE, "rb") as f:
            vectorstore = pickle.load(f)

    # ✅ Step 2: Ensure an index is loaded
    if not vectorstore:
        return render_template("index.html", answer="⚠️ Please upload a document first.")

    # ✅ Step 3: Ensure user entered a question
    if not query:
        return render_template("index.html", answer="⚠️ Please enter a question.")

    # ✅ Step 4: Retrieve relevant context from document
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    results = retriever.get_relevant_documents(query)
    context = "\n\n".join([r.page_content for r in results])

    # ✅ Step 5: Create structured prompt
    final_prompt = f"""
You are an intelligent document assistant.
Use the following context to answer the user's question clearly and in well-formatted text.

### Context:
{context}

### Question:
{query}

### Instructions:
- Provide a detailed yet concise answer.
- Use bullet points or numbered lists when helpful.
- Highlight key terms using **bold**.
"""

    # ✅ Step 6: Generate formatted response
    try:
        response = model.generate_content(final_prompt)
        raw_answer = response.text
        formatted_answer = markdown(raw_answer)

        answer = f"""
        <div style='
            font-family: "Segoe UI", sans-serif;
            background-color: #f9fafb;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            line-height: 1.6;
            font-size: 16px;
        '>
        {formatted_answer}
        </div>
        """
    except Exception as e:
        answer = f"⚠️ Error generating response: {str(e)}"

    # ✅ Render formatted answer safely
    return render_template("index.html", answer=answer, question=query)

if __name__ == "__main__":
    app.run(debug=True)
