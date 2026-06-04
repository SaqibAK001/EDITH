import uuid
import tempfile

from rest_framework.decorators import api_view
from rest_framework.response import Response

from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

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

    # ==========================
    # NEW DOCUMENT UPLOAD
    # ==========================

    if uploaded_file:

        # Create temporary PDF
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)

            temp_path = temp_file.name

        try:

            # Read PDF
            loader = PyPDFLoader(temp_path)

            pages = loader.load()

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            docs = splitter.split_documents(
                pages
            )

            # Build FAISS index
            vectorstore = FAISS.from_documents(
                docs,
                get_embedding_model
            )

            # Generate unique session
            session_id = str(
                uuid.uuid4()
            )

            create_session(
                session_id,
                vectorstore
            )

            return Response({
                "session_id": session_id,
                "message": "Document uploaded successfully"
            })

        finally:

            # Delete temporary PDF immediately
            import os

            if os.path.exists(temp_path):
                os.remove(temp_path)

    # ==========================
    # ASK QUESTION
    # ==========================

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