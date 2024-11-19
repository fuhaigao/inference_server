# Variables
FRONTEND_DIR=frontend
BINARY_NAME=inference_server

# Default target
all: up

# Build the frontend and backend
build:
	cd $(FRONTEND_DIR) && npm install && npm run build
	cargo build

# Clean the project dependencies
clean:
	cargo clean

# Rebuild everything
rebuild: clean build

# Compile and start the server
up: build
	cargo run --bin $(BINARY_NAME)

# restart the application
restart: clean up

# Run the application without re-compiling
start:
	cargo run --bin $(BINARY_NAME)
