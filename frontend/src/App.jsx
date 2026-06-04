import { useState } from "react";
import api from "./api";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

function App() {
  const [files, setFiles] = useState([]);
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [darkMode, setDarkMode] = useState(true);
  const [uploadStatus, setUploadStatus] = useState("");

  const uploadDocument = async () => {
    if (!files.length) {
      setUploadStatus("Please select a document.");
      return;
    }

    const formData = new FormData();
    formData.append("file", files[0]);

    try {
      setUploadStatus("Uploading document...");

      const res = await api.post("/ask/", formData);

      setSessionId(res.data.session_id);

      setUploadStatus("✅ Document uploaded successfully.");
    } catch (err) {
      console.error(err);
      setUploadStatus("❌ Upload failed.");
    }
  };

  const askQuestion = async () => {
    if (!sessionId) {
      setAnswer("⚠️ Please upload a document first.");
      return;
    }

    if (!query.trim()) {
      setAnswer("⚠️ Please enter a question.");
      return;
    }

    try {
      const res = await api.post("/ask/", {
        query: query,
        session_id: sessionId,
      });

      setAnswer(res.data.answer);
    } catch (err) {
      console.error(err);
      setAnswer("❌ Error connecting to backend.");
    }
  };

  const toggleTheme = () => {
    setDarkMode(!darkMode);
  };

  return (
    <div className={darkMode ? "app dark" : "app"}>
      <div className="container">
        <button className="theme-toggle" onClick={toggleTheme}>
          {darkMode ? "☀️" : "🌙"}
        </button>

        <h1>EDITH</h1>
        <p>Your AI Document Analyzer</p>

        <input
          type="file"
          accept=".pdf"
          onChange={(e) => setFiles(Array.from(e.target.files))}
        />

        <button onClick={uploadDocument}>
          Upload Document
        </button>

        {uploadStatus && (
          <p style={{ marginTop: "10px" }}>
            {uploadStatus}
          </p>
        )}

        <input
          type="text"
          placeholder="Ask a question..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />

        <button onClick={askQuestion}>
          Ask Question
        </button>

        {answer && (
          <div className="answer">
            <h2>💡 Answer</h2>

            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
            >
              {answer}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;