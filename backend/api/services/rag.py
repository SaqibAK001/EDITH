import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# =========================
# CONFIG
# =========================

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

def get_model():
    return genai.GenerativeModel(
        "models/gemini-2.5-flash"
    )

def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

# =========================
# SESSION STORAGE
# =========================

SESSIONS = {}

MAX_QUESTIONS = 15
SESSION_TIMEOUT = 1800  # 30 minutes


# =========================
# SESSION MANAGEMENT
# =========================

def cleanup_sessions():
    now = time.time()

    expired = []

    for sid, session in SESSIONS.items():

        if (
            now - session["created"] > SESSION_TIMEOUT
            or session["questions"] >= MAX_QUESTIONS
        ):
            expired.append(sid)

    for sid in expired:
        del SESSIONS[sid]


def create_session(session_id, vectorstore):

    cleanup_sessions()

    SESSIONS[session_id] = {
        "vectorstore": vectorstore,
        "questions": 0,
        "created": time.time()
    }


def get_session(session_id):

    cleanup_sessions()

    return SESSIONS.get(session_id)


# =========================
# QUERY PROCESSING
# =========================

def process_query(session_id, query):

    session = get_session(session_id)

    if not session:
        return (
            "⚠️ Session expired. "
            "Please upload the document again."
        )

    if not query.strip():
        return "⚠️ Please enter a question."

    session["questions"] += 1

    vectorstore = session["vectorstore"]

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    results = retriever.get_relevant_documents(
        query
    )

    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    final_prompt = f"""
You are an intelligent document assistant.

### Context:
{context}

### Question:
{query}

### Instructions:
- Format your answer in Markdown.
- Use headings (##) for major sections.
- Use bullet points for lists.
- Use **bold** for important concepts.
- Use tables when useful.
- Generate valid GitHub-Flavored Markdown (GFM).
- Keep the answer structured and readable.
- Base the answer ONLY on the provided context.
"""

    try:

        response = get_model.generate_content(
            final_prompt
        )

        return response.text

    except Exception as e:

        return (
            f"⚠️ Error generating response: {str(e)}"
        )