import ReactMarkdown from "react-markdown";

function ResponseBox({ answer }) {
  return (
    <div>
      <ReactMarkdown>
        {answer}
      </ReactMarkdown>
    </div>
  );
}

export default ResponseBox;