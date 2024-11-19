# LLM Serving (Implemented in Rust)

## Prerequirements
1. Ensure you have **Rust** installed. You can check by running `rustc --version`.
2. Ensure **nodejs** is installed. You can check by running `npm --version`.

## How to Run

### Approach 1: using Makefile (requires make installed)
1. Run `make up` to build and start the application.
2. Optionally, run `make restart` to clean, rebuild, and restart the application.
3. Optionally, run `make start` to start the application without rebuilding.

### Approach 2: using cargo and npm
1. Run `cd frontend && npm install && npm run build && cd ..` to build the frontend.
2. Run `cargo build` to build the backend.
3. Run `cargo run --bin inference_server` to start the application.
