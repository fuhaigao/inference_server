import axios from "axios";

const apiClient = axios.create({
  baseURL: "http://localhost:8080", // Update to your Rust server's base URL
});

export interface TopResult {
  item: string;
  score: number;
}

export const generateText = async (input: string): Promise<string> => {
  const requestBody = {
    prompt: input,
    max_length: 20,
  };

  const response = await apiClient.post("/generate_text", requestBody);
  return response.data.generated_text;
};

export const findSimilar = async (input: string): Promise<TopResult[]> => {
  const requestBody = {
    text: input,
    num_results: 5,
  };

  const response = await apiClient.post("/find_similar", requestBody);
  return response.data.top_results; // Return the array of objects
};
