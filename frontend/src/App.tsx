import React, { useState } from "react";
import { generateText, findSimilar, TopResult } from "./api";

function App() {
  const [input, setInput] = useState("");
  const [output, setOutput] = useState<string | TopResult[]>("");
  const [mode, setMode] = useState<"generate_text" | "find_similar">(
    "generate_text",
  );
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  const handleSubmit = async () => {
    if (input.trim() === "") return;
    setOutput("Processing...");
    try {
      if (mode === "generate_text") {
        const result = await generateText(input);
        setOutput(result);
      } else {
        const results = await findSimilar(input);
        setOutput(results);
      }
    } catch (error) {
      setOutput("An error occurred while processing your request.");
      console.error(error);
    }
  };

  const handleModeChange = (newMode: "generate_text" | "find_similar") => {
    setMode(newMode);
    setInput("");
    setOutput("");
    setUploadedFile(null);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    setUploadedFile(file);
  };

  const renderOutput = () => {
    if (Array.isArray(output)) {
      return (
        <ul>
          {output.map((result, index) => (
            <li key={index}>
              <strong>{result.item}</strong> - Score: {result.score.toFixed(2)}
            </li>
          ))}
        </ul>
      );
    }
    return (
      <pre style={{ background: "#f4f4f4", padding: "10px" }}>{output}</pre>
    );
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h1>LLM Serving (Rust) Demo</h1>
      <div>
        <label>
          Select Mode:
          <select
            value={mode}
            onChange={(e) =>
              handleModeChange(
                e.target.value as "generate_text" | "find_similar",
              )
            }
          >
            <option value="generate_text">Generate Text (LLAMA Model)</option>
            <option value="find_similar">Find Similar (BERT Model)</option>
          </select>
        </label>
      </div>
      {mode === "find_similar" && (
        <div style={{ margin: "10px 0" }}>
          <label>
            Upload CSV File:
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              style={{ marginLeft: "10px" }}
            />
          </label>
          {uploadedFile && (
            <div style={{ marginTop: "10px" }}>
              <strong>Uploaded File:</strong> {uploadedFile.name}
            </div>
          )}
        </div>
      )}
      <div style={{ margin: "20px 0" }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Enter your input here..."
          rows={5}
          style={{ width: "100%" }}
          // disabled={mode === "find_similar" && !!uploadedFile}
        />
      </div>
      <button onClick={handleSubmit} disabled={input === ""}>
        Submit
      </button>
      <div style={{ marginTop: "20px" }}>
        <h3>Output:</h3>
        {renderOutput()}
      </div>
    </div>
  );
}

export default App;
