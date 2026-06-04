function FileUpload({ setFiles }) {
  return (
    <input
      type="file"
      multiple
      onChange={(e) => setFiles(Array.from(e.target.files))}
    />
  );
}

export default FileUpload;