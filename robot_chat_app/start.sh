#!/bin/bash

# Startup script for robot chat application

echo "Starting robot control chat application..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠ Warning: .env file not found"
    echo "Creating .env from env.example..."
    cp env.example .env
    echo ""
    echo "Please edit .env and add your OPENAI_API_KEY"
    echo "Then run this script again"
    exit 1
fi

# Check if OPENAI_API_KEY is set
source .env
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    echo "⚠ Error: OPENAI_API_KEY not set in .env file"
    echo "Please edit .env and add your OpenAI API key"
    exit 1
fi

echo "Starting Docker containers..."
docker-compose up --build

echo "Application stopped"

