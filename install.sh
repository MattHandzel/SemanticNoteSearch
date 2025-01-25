#! /usr/bin/env bash
# ChatGPT'd

# Function to check if Docker is installed
check_docker() {
    if command -v docker &> /dev/null; then
        echo "Docker is installed."
        return 0
    else
        echo "Docker is not installed. Please install Docker and try again."
        return 1
    fi
}

# Function to check if Qdrant image is already pulled
check_qdrant_image() {
    if docker image inspect qdrant/qdrant &> /dev/null; then
        echo "Qdrant image is already pulled."
        return 0
    else
        echo "Qdrant image is not pulled. Pulling now..."
        return 1
    fi
}

# Function to pull Qdrant image
pull_qdrant() {
    docker pull qdrant/qdrant
    if [ $? -eq 0 ]; then
        echo "Qdrant image pulled successfully."
    else
        echo "Failed to pull Qdrant image. Please check your internet connection and try again."
        exit 1
    fi
}

# Main installation process
main() {
    echo "Starting installation process..."

    # Check if Docker is installed
    check_docker
    if [ $? -ne 0 ]; then
        exit 1
    fi

    # Check if Qdrant image is already pulled
    check_qdrant_image
    if [ $? -ne 0 ]; then
        pull_qdrant
    fi

    # Add any additional installation steps here
    echo "Installation completed successfully."
}

# Run the main function
main
