# Variables
FRONTEND_DIR=frontend
BINARY_NAME=inference_server

# Default target
all: up

# Build the frontend and backend
build:
	cd $(FRONTEND_DIR) && npm run build
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
