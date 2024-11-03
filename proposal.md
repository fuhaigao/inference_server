# Project Proposal: Large Language Model Serving

Fuhai Gao
1003887065

## Motivation

Large language models (LLMs) have gained significant attention for their ability to process and generate natural language, enabling applications across industries in fields like customer service, content creation, and knowledge management. However, most existing LLM-serving solutions are implemented in Python, which, while widely adopted, can impose performance limitations due to its interpreted nature and runtime dependencies.

This project explores building a high-performance LLM inference server using Rust, a systems programming language known for speed, safety, and low-level control. By leveraging Rust’s efficiency and the inference capabilities of libraries like Candle, this project aims to create a reliable, customizable inference server that offers significant advantages over traditional Python-based solutions.

### Why This Project is Meaningful

Existing LLM-serving applications, while providing API endpoints that are easy to integrate with, do not offer users the flexibility to manage different models or implement custom features. Additionally, most inference servers are Python-based, introducing runtime overhead due to Python's interpreted nature. Implementing an inference server using Rust has the potential to improve performance while also providing developers with greater flexibility.

This project also aims to contribute to the Rust ecosystem by offering a native, high-performance inference solution specifically tailored for LLMs, demonstrating Rust's potential in a field typically dominated by Python.

### Comparison: Rust-Based Custom Solution vs. Existing Python-Based Inference Services

| Feature                     | **Rust-Based Custom Inference**                       | **Python-Based Services**                          |
|-----------------------------|------------------------------------------------------|---------------------------------------------------|
| **Performance Potential**   | High performance with near-native speed              | High but limited by Python’s runtime overhead      |
| **Customizability**         | Full control over the inference pipeline             | Constrained by pre-built frameworks and abstractions |
| **Concurrency and Multithreading** | True parallelism without a GIL                   | Limited by the Global Interpreter Lock (GIL) in Python |
| **Memory Management**       | Explicit control, no garbage collection pauses       | Automatic but introduces potential unpredictability |
| **Deployment**              | Lightweight, efficient binaries ideal for serverless and edge deployments | Higher memory and runtime footprint, making deployment more complex |
| **Use Case Fit**            | Real-time, high-performance applications with resource constraints | Prototyping, research, and applications where Python’s overhead is manageable |

A Rust-based solution addresses several limitations inherent in Python-based LLM-serving solutions, such as high memory usage, slower startup times, and constraints in multithreading. With this project, Rust can demonstrate a new paradigm for LLM-serving, offering an efficient alternative that meets the demands of modern, resource-intensive applications.

## Objectives and Key Features

### Objective

The main objective is to develop an LLM inference server using Rust and Candle that efficiently handles requests, manages multiple models, and provides streaming support. The server will feature an accessible API layer for flexible deployment in various applications, enabling developers to integrate advanced LLM capabilities with ease.

### Key Features

To achieve this objective, the following key features will be implemented:

1. **Multi-Model Management**

    The inference server will be capable of loading and managing multiple large language models simultaneously. This feature will enable users to serve multiple models or versions based on different application requirements, providing versatility and reducing response times by preloading models.

2. **API Endpoints**

    A core feature of the server is an API layer that exposes multiple endpoints, allowing clients to send inputs to the models and receive processed outputs. This API will support several types of requests, including simple text-based inputs for standard LLM tasks and more complex inputs for optional features like image generation.

3. **Streaming Support**

    Many LLM applications benefit from real-time response streaming, especially for large outputs or ongoing interactions. The server will incorporate streaming capabilities, allowing responses to be delivered progressively as they are generated. This feature creates a more interactive experience for applications that rely on real-time outputs.

4. **Basic Chat Interface**

    To demonstrate the server’s capabilities, a basic chat interface will be created. This interface will allow users to interact with the models through simple, conversational prompts, simulating a chat-based LLM experience. The chat interface will showcase the model’s ability to handle natural language queries and offer an accessible entry point for user interaction.

5. **Optional Advanced Features**

    Beyond the core functionality, the server will offer several advanced features to showcase its ability to process input with various models:

    - **Embedding Search Using BERT**

        This feature will allow users to perform similarity-based searches across a dataset. A preloaded CSV file of topics will be converted into embeddings, and users can submit text queries to find the most relevant topics based on similarity, making it suitable for search and recommendation systems.

    - **Text-to-Image Generation**

        By integrating a model capable of generating images from text, this feature enables creative applications like visual content creation or design prototyping. Users can submit descriptive text and receive generated images.

    - **Question-Answering Model**

        A conversational AI feature, similar to systems like ChatGPT, will be included to answer user-submitted questions. This feature would handle tasks such as knowledge retrieval or interactive assistance, allowing users to ask questions and receive relevant answers from the model.

    - **Additional Model-Based Capabilities**

        Investigate other interesting open-source models that are compatible with Candle, giving users the option to experiment with different functionalities, such as summarization or text classification, and further extending the server’s applicability.


## Tentative Plan

The project will be completed by one developer working part-time, dedicating 1-2 hours daily over a 7-week period. The plan is organized into phases to ensure structured progress, with a focus on completing foundational components early and incorporating additional features based on remaining time.

1. **Weeks 1-2: Project Setup and Research**
    - Initialize the Rust project, set up dependencies, and finalize the backend framework (either Rocket or Actix Web).
    - Conduct a thorough review of Candle or Mistral.rs, establishing a foundational understanding of model inference setup.

2. **Weeks 3-4: Core Server Implementation**
    - Develop the model-loading system, enabling the server to preload and manage multiple models.
    - Build the API endpoints to handle model interaction requests. Ensure API requests can be successfully handled.
    - Start connecting the API endpoints to the LLM processing. Parse user input and feed them into loaded models. Then parse the result from the models and return a well-formatted response to API users.

3. **Weeks 5-6: Streaming and Optional Feature Integration**
    - Implement streaming capabilities for inference responses.
    - Begin integration of optional features such as embedding search and text-to-image generation, prioritizing features based on complexity and demand.
    - Build a simple interface to demo the features.

4. **Week 7: Testing and Finalization**
    - Perform end-to-end testing on all implemented features to ensure the frontend successfully sends API requests, receives responses, and displays them accordingly.
    - Prepare for the video demo.

## Conclusion

The Large Language Model Serving project aims to provide a Rust-based inference solution that addresses existing limitations in Python-based systems. By focusing on customizability and streaming support, this project will establish Rust as an alternative for high-performance LLM applications, contributing to the growing ecosystem of Rust in machine learning and real-time inference.
