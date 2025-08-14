#!/bin/bash


clear


GREEN='\033[0;32m'

RED='\033[0;31m'

YELLOW='\033[0;33m'

BLUE='\033[0;34m'

CYAN='\033[0;36m'

BOLD='\033[1m'

PURPLE='\033[0;35m'

NC='\033[0m' # No Color



echo -e "${BLUE}"

echo "       █████╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███████╗ ██████╗██╗  ██╗     ██████╗ ██████╗ ███████╗"

echo "      ██╔══██╗██╔══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔════╝██║  ██║    ██╔════╝██╔═══██╗██╔════╝"

echo "      ███████║██║  ██║██║   ██║███████║██╔██╗ ██║   ██║   █████╗  ██║     ███████║    ██║     ██║   ██║█████╗  "

echo "      ██╔══██║██║  ██║╚██╗ ██╔╝██╔══██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██╔══██║    ██║     ██║   ██║██╔══╝  "

echo "      ██║  ██║██████╔╝ ╚████╔╝ ██║  ██║██║ ╚████║   ██║   ███████╗╚██████╗██║  ██║    ╚██████╗╚██████╔╝███████╗"

echo "      ╚═╝  ╚═╝╚═════╝   ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝     ╚═════╝ ╚═════╝ ╚══════╝"

echo -e "${WHITE}                                  Center of Excellence${NC}"

echo

echo -e "${CYAN}  This may take a moment...${NC}"

echo

sleep 7

set -e

# Load environment variables
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

echo "Using model: $MODEL_NAME"
echo "Compose file: $COMPOSE_FILE_PATH"

# Start Docker containers
# Run docker-compose using either plugin or legacy binary
echo "Running docker-compose"
if [ -f "$COMPOSE_FILE_PATH" ]; then
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$COMPOSE_FILE_PATH" up --build --force-recreate -d
    elif docker compose version &> /dev/null; then
        docker compose -f "$COMPOSE_FILE_PATH" up --build --force-recreate -d
    else
        echo "Neither docker-compose nor 'docker compose' found!"
        exit 1
    fi
    echo "Docker Compose started successfully."
else
    echo "docker-compose.yml not found at $COMPOSE_FILE_PATH"
    exit 1
fi

echo "All setup complete"

# Connect to container
echo "Connecting to container..."
docker exec -it Deepseek-R1-1.5B-Langchain-AI-Agent-RAG-on-NVIDIA-Jetson bash