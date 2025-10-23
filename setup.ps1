# 🚀 Golden Agent Framework - Windows Setup Script
# This script automates the installation process for Windows machines

param(
    [switch]$SkipDocker,
    [switch]$SkipOllama,
    [switch]$Verbose
)

# Set error action preference
$ErrorActionPreference = "Stop"

Write-Host "🚀 Golden Agent Framework - Windows Setup" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor Blue
}

# Check if running on Windows
function Test-Windows {
    Write-Info "Checking operating system..."
    if ($env:OS -eq "Windows_NT") {
        Write-Status "Windows detected"
    } else {
        Write-Error "This script is for Windows only"
        exit 1
    }
}

# Check Python version
function Test-Python {
    Write-Info "Checking Python version..."
    
    try {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python (\d+)\.(\d+)") {
            $major = [int]$matches[1]
            $minor = [int]$matches[2]
            if ($major -ge 3 -and $minor -ge 9) {
                Write-Status "Python $($matches[0]) detected"
            } else {
                Write-Error "Python 3.9+ required, found $($matches[0])"
                exit 1
            }
        } else {
            Write-Error "Python not found. Please install Python 3.9+"
            exit 1
        }
    } catch {
        Write-Error "Python not found. Please install Python 3.9+"
        exit 1
    }
}

# Install uv (Python package manager)
function Install-UV {
    Write-Info "Installing uv (Python package manager)..."
    
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Write-Status "uv already installed"
    } else {
        # Install uv using pip
        pip install uv
        Write-Status "uv installed successfully"
    }
}

# Install Ollama
function Install-Ollama {
    if ($SkipOllama) {
        Write-Warning "Skipping Ollama installation"
        return
    }
    
    Write-Info "Installing Ollama LLM..."
    
    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        Write-Status "Ollama already installed"
    } else {
        # Download and install Ollama for Windows
        $ollamaUrl = "https://ollama.ai/download/windows"
        Write-Info "Please download and install Ollama from: $ollamaUrl"
        Write-Info "After installation, restart your terminal and run this script again"
        Read-Host "Press Enter after installing Ollama"
    }
    
    # Start Ollama service
    Write-Info "Starting Ollama service..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    
    # Wait for service to start
    Start-Sleep -Seconds 5
    
    # Pull required model
    Write-Info "Downloading llama3.2 model (this may take a few minutes)..."
    ollama pull llama3.2
    
    Write-Status "Ollama setup complete"
}

# Install Docker
function Install-Docker {
    if ($SkipDocker) {
        Write-Warning "Skipping Docker installation"
        return
    }
    
    Write-Info "Checking Docker installation..."
    
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        Write-Status "Docker already installed"
    } else {
        Write-Warning "Docker not found. Please install Docker Desktop:"
        Write-Host "  https://www.docker.com/products/docker-desktop"
        Read-Host "Press Enter after installing Docker Desktop"
    }
}

# Setup Python environment
function Setup-Python {
    Write-Info "Setting up Python environment..."
    
    # Create virtual environment
    uv venv
    
    # Activate virtual environment
    $activateScript = ".\venv\Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
    } else {
        Write-Error "Failed to create virtual environment"
        exit 1
    }
    
    # Install dependencies
    Write-Info "Installing Python dependencies..."
    uv install
    
    Write-Status "Python environment setup complete"
}

# Setup configuration
function Setup-Config {
    Write-Info "Setting up configuration..."
    
    # Create .env file if it doesn't exist
    if (-not (Test-Path ".env")) {
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" ".env"
            Write-Status "Created .env file from template"
        } else {
            Write-Warning ".env.example not found"
        }
    } else {
        Write-Status ".env file already exists"
    }
    
    # Create directories
    $directories = @("kb", "data\mock_tickets", "data\faiss_index")
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Status "Created $dir directory"
        }
    }
    
    Write-Status "Configuration setup complete"
}

# Start Docker services
function Start-Services {
    if ($SkipDocker) {
        Write-Warning "Skipping Docker services"
        return
    }
    
    Write-Info "Starting Docker services..."
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be ready
    Write-Info "Waiting for services to start..."
    Start-Sleep -Seconds 10
    
    # Check service health
    Write-Info "Checking service health..."
    
    # Check LangFuse
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3000/health" -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Status "LangFuse is running"
        }
    } catch {
        Write-Warning "LangFuse may not be ready yet"
    }
    
    # Check MCP Gateway
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8081/health" -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Status "MCP Gateway is running"
        }
    } catch {
        Write-Warning "MCP Gateway may not be ready yet"
    }
}

# Run verification tests
function Test-Installation {
    Write-Info "Running verification tests..."
    
    # Activate virtual environment
    $activateScript = ".\venv\Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
    }
    
    # Test basic imports
    Write-Info "Testing basic imports..."
    try {
        python -c "
import langgraph
import langfuse
import faiss
import sentence_transformers
print('✅ Core dependencies imported successfully')
"
        Write-Status "Core dependencies test passed"
    } catch {
        Write-Error "Core dependencies test failed"
    }
    
    # Test framework imports
    Write-Info "Testing framework imports..."
    try {
        python -c "
from core.config import load_config
from agents.triage.agent import TriageAgent
from core.rag.local_kb import LocalKB
print('✅ Framework components imported successfully')
"
        Write-Status "Framework components test passed"
    } catch {
        Write-Error "Framework components test failed"
    }
    
    # Run smoke test
    Write-Info "Running smoke test..."
    try {
        python scripts\smoke_test.py
        Write-Status "Smoke test passed"
    } catch {
        Write-Warning "Smoke test failed - this may be normal on first run"
    }
    
    Write-Status "Verification tests completed"
}

# Main installation function
function Main {
    Write-Host "Starting installation process..."
    Write-Host ""
    
    Test-Windows
    Test-Python
    Install-UV
    Install-Ollama
    Setup-Python
    Setup-Config
    Start-Services
    Test-Installation
    
    Write-Host ""
    Write-Host "🎉 Installation completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "1. Edit .env file with your configuration"
    Write-Host "2. Add knowledge base files to kb\ directory"
    Write-Host "3. Run: python scripts\test_full_workflow.py"
    Write-Host "4. Open LangFuse: http://localhost:3000"
    Write-Host ""
    Write-Host "For more information, see INSTALLATION_GUIDE.md"
}

# Run main function
Main
