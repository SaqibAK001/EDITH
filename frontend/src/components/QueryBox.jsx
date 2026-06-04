function QueryBox({ query, setQuery }) {
  return (
    <textarea
      value={query}
      onChange={(e) => setQuery(e.target.value)}
      placeholder="Ask EDITH..."
    />
  );
}

export default QueryBox;