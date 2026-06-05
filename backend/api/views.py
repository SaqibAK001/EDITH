import uuid
import tempfile
import os

from rest_framework.decorators import api_view
from rest_framework.response import Response

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from .services.rag import (
    create_session,
    process_query,
    get_embedding_model
)


@api_view(["POST"])
def ask_document(request):

    query = request.data.get("query", "")
    session_id = request.data.get("session_id")

    uploaded_file = request.FILES.get("file")

    # ==================================
    # NEW DOCUMENT UPLOAD
    # ==================================

    if uploaded_file:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)

            temp_path = temp_file.name

        try:

            # Load PDF
            loader = PyPDFLoader(temp_path)
            pages = loader.load()

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            docs = splitter.split_documents(pages)

            # Create embeddings using Google API
            embeddings = get_embedding_model()

            # Build FAISS index
            vectorstore = FAISS.from_documents(
                docs,
                embeddings
            )

            # Create session
            session_id = str(uuid.uuid4())

            create_session(
                session_id,
                vectorstore
            )

            return Response({
                "session_id": session_id,
                "message": "Document uploaded successfully"
            })

        except Exception as e:

            return Response({
                "error": str(e)
            }, status=500)

        finally:

            if os.path.exists(temp_path):
                os.remove(temp_path)

    # ==================================
    # ASK QUESTION
    # ==================================

    if not session_id:

        return Response({
            "answer": "⚠️ Upload a document first."
        })

    answer = process_query(
        session_id,
        query
    )

    return Response({
        "answer": answer
    })