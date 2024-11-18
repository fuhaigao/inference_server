import React, { useEffect, useState } from "react";
import { generateText, findSimilar, TopResult } from "./api";

function App() {
  const [input, setInput] = useState("");
  const [output, setOutput] = useState<string | TopResult[]>("");
  const [mode, setMode] = useState<
    "generate_text" | "find_similar" | "generate_text_stream"
  >("generate_text");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  useEffect(() => {
    console.log("output", output);
  }, [output]);

  // const handleSubmit = async () => {
  //   if (input.trim() === "") return;
  //   setOutput("Processing...");
  //   try {
  //     if (mode === "generate_text_stream") {
  //       setOutput(""); // Clear previous output

  //       const eventSource = new EventSource(
  //         `http://localhost:8080/generate_text_stream?prompt=${encodeURIComponent(input)}&max_length=20&timestamp=${Date.now()}`,
  //       );

  //       eventSource.onmessage = (event) => {
  //         console.log("Received event:", event.data);
  //         if (event.data === "EOS") {
  //           console.log("End of stream");
  //           eventSource.close();
  //         } else {
  //           setOutput((prevOutput) => prevOutput + event.data);
  //         }
  //       };

  //       eventSource.onerror = (error) => {
  //         console.error("EventSource failed:", error);
  //         eventSource.close();
  //         setOutput("An error occurred while streaming the text.");
  //       };
  //     } else if (mode === "generate_text") {
  //       const result = await generateText(input);
  //       setOutput(result);
  //     } else {
  //       const results = await findSimilar(input);
  //       setOutput(results);
  //     }
  //   } catch (error) {
  //     setOutput("An error occurred while processing your request.");
  //     console.error(error);
  //   }
  // };

  const handleSubmit = async () => {
    if (input.trim() === "") return;
    setOutput("Processing...");
    try {
      if (mode === "generate_text_stream") {
        setOutput(""); // Clear previous output

        // Use fetch for POST request
        const response = await fetch(
          "http://localhost:8080/generate_text_stream",
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              prompt: input,
              max_length: 20,
            }),
          },
        );

        if (!response.ok) {
          console.error(
            "Failed to connect to the server:",
            response.statusText,
          );
          setOutput("An error occurred while connecting to the server.");
          return;
        }

        // Check if response body is null
        if (!response.body) {
          console.error("Response body is null");
          setOutput("An error occurred while processing the stream.");
          return;
        }

        // Process the streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          let boundary;
          while ((boundary = buffer.indexOf("\n\n")) !== -1) {
            const chunk = buffer.slice(0, boundary); // Extract the complete message
            buffer = buffer.slice(boundary + 2); // Remove the processed part

            if (chunk.startsWith("data: ")) {
              const data = chunk.slice(6); // Strip the "data: " prefix
              console.log("Received event:", data);

              if (data === "EOS") {
                console.log("End of stream");
                setOutput((prevOutput) => prevOutput + "\n[End of Stream]");
                return; // Close the connection
              } else {
                setOutput((prevOutput) => prevOutput + data);
              }
            }
          }
        }
      } else if (mode === "generate_text") {
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

  const handleModeChange = (
    newMode: "generate_text" | "find_similar" | "generate_text_stream",
  ) => {
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
                e.target.value as
                  | "generate_text"
                  | "find_similar"
                  | "generate_text_stream",
              )
            }
          >
            <option value="generate_text">Generate Text (LLAMA Model)</option>
            <option value="generate_text_stream">
              Generate Text Stream (LLAMA Model)
            </option>
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
          disabled={mode === "find_similar" && !!uploadedFile}
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
