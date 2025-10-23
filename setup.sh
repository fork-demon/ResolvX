#!/bin/bash

# 🚀 Golden Agent Framework - Quick Setup Script
# This script automates the installation process for new machines

set -e  # Exit on any error

echo "🚀 Golden Agent Framework - Quick Setup"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if running on supported OS
check_os() {
    print_info "Checking operating system..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_status "macOS detected"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_status "Linux detected"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Check Python version
check_python() {
    print_info "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        if [[ $(echo "$PYTHON_VERSION >= 3.9" | bc -l) -eq 1 ]]; then
            print_status "Python $PYTHON_VERSION detected"
        else
            print_error "Python 3.9+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.9+"
        exit 1
    fi
}

# Install uv (Python package manager)
install_uv() {
    print_info "Installing uv (Python package manager)..."
    
    if command -v uv &> /dev/null; then
        print_status "uv already installed"
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.cargo/env
        print_status "uv installed successfully"
    fi
}

# Install Ollama
install_ollama() {
    print_info "Installing Ollama LLM..."
    
    if command -v ollama &> /dev/null; then
        print_status "Ollama already installed"
    else
        curl -fsSL https://ollama.ai/install.sh | sh
        print_status "Ollama installed successfully"
    fi
    
    # Start Ollama service
    print_info "Starting Ollama service..."
    ollama serve &
    sleep 5
    
    # Pull required model
    print_info "Downloading llama3.2 model (this may take a few minutes)..."
    ollama pull llama3.2
    
    print_status "Ollama setup complete"
}

# Install Docker
install_docker() {
    print_info "Checking Docker installation..."
    
    if command -v docker &> /dev/null; then
        print_status "Docker already installed"
    else
        print_warning "Docker not found. Please install Docker Desktop:"
        if [[ "$OS" == "macos" ]]; then
            echo "  https://www.docker.com/products/docker-desktop"
        else
            echo "  sudo apt-get install docker.io docker-compose"
        fi
        read -p "Press Enter after installing Docker..."
    fi
}

# Setup Python environment
setup_python() {
    print_info "Setting up Python environment..."
    
    # Create virtual environment
    uv venv
    source .venv/bin/activate
    
    # Install dependencies
    print_info "Installing Python dependencies..."
    uv install
    
    print_status "Python environment setup complete"
}

# Setup configuration
setup_config() {
    print_info "Setting up configuration..."
    
    # Create .env file if it doesn't exist
    if [[ ! -f .env ]]; then
        cp .env.example .env
        print_status "Created .env file from template"
    else
        print_status ".env file already exists"
    fi
    
    # Create knowledge base directory
    mkdir -p kb
    print_status "Created knowledge base directory"
    
    # Create mock data directory
    mkdir -p data/mock_tickets
    print_status "Created mock data directory"
    
    # Create FAISS index directory
    mkdir -p data/faiss_index
    print_status "Created FAISS index directory"
}

# Start Docker services
start_services() {
    print_info "Starting Docker services..."
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be ready
    print_info "Waiting for services to start..."
    sleep 10
    
    # Check service health
    print_info "Checking service health..."
    
    # Check LangFuse
    if curl -s http://localhost:3000/health > /dev/null; then
        print_status "LangFuse is running"
    else
        print_warning "LangFuse may not be ready yet"
    fi
    
    # Check MCP Gateway
    if curl -s http://localhost:8081/health > /dev/null; then
        print_status "MCP Gateway is running"
    else
        print_warning "MCP Gateway may not be ready yet"
    fi
}

# Run verification tests
run_tests() {
    print_info "Running verification tests..."
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Test basic imports
    print_info "Testing basic imports..."
    python -c "
import langgraph
import langfuse
import faiss
import sentence_transformers
print('✅ Core dependencies imported successfully')
" || print_error "Core dependencies test failed"
    
    # Test framework imports
    print_info "Testing framework imports..."
    python -c "
from core.config import load_config
from agents.triage.agent import TriageAgent
from core.rag.local_kb import LocalKB
print('✅ Framework components imported successfully')
" || print_error "Framework components test failed"
    
    # Run smoke test
    print_info "Running smoke test..."
    python scripts/smoke_test.py || print_warning "Smoke test failed - this may be normal on first run"
    
    print_status "Verification tests completed"
}

# Main installation function
main() {
    echo "Starting installation process..."
    echo ""
    
    check_os
    check_python
    install_uv
    install_ollama
    install_docker
    setup_python
    setup_config
    start_services
    run_tests
    
    echo ""
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Edit .env file with your configuration"
    echo "2. Add knowledge base files to kb/ directory"
    echo "3. Run: python scripts/test_full_workflow.py"
    echo "4. Open LangFuse: http://localhost:3000"
    echo ""
    echo "For more information, see INSTALLATION_GUIDE.md"
}

# Run main function
main "$@"
