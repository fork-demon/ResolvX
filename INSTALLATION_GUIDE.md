# 🚀 Golden Agent Framework - Installation Guide

## 📋 **System Requirements**

### **Minimum Requirements**
- **OS**: macOS 12+, Ubuntu 20.04+, Windows 10+
- **Python**: 3.9+ (recommended: 3.11)
- **RAM**: 8GB+ (recommended: 16GB)
- **Storage**: 10GB+ free space
- **Docker**: 20.10+ (for services)

### **Recommended Setup**
- **OS**: macOS 14+ or Ubuntu 22.04+
- **Python**: 3.11
- **RAM**: 16GB+
- **Storage**: 20GB+ free space
- **Docker**: Latest version

---

## 🛠️ **Installation Methods**

### **Method 1: Quick Setup with uv (Recommended)**

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/your-org/golden-agent-framework.git
cd golden-agent-framework

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies
uv install

# Verify installation
python -c "import langgraph, langfuse, faiss; print('✅ All dependencies installed')"
```

### **Method 2: Traditional pip Installation**

```bash
# Clone repository
git clone https://github.com/your-org/golden-agent-framework.git
cd golden-agent-framework

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import langgraph, langfuse, faiss; print('✅ All dependencies installed')"
```

### **Method 3: Docker Development Environment**

```bash
# Clone repository
git clone https://github.com/your-org/golden-agent-framework.git
cd golden-agent-framework

# Build development container
docker-compose -f docker-compose.dev.yml up -d

# Access container
docker-compose -f docker-compose.dev.yml exec app bash

# Install dependencies inside container
pip install -r requirements.txt
```

---

## 🔧 **Service Dependencies**

### **1. Ollama LLM (Required)**

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull required model
ollama pull llama3.2

# Verify installation
ollama list
# Should show: llama3.2
```

### **2. Docker Services (Required)**

```bash
# Install Docker
# macOS: Download from https://www.docker.com/products/docker-desktop
# Ubuntu: sudo apt-get install docker.io docker-compose
# Windows: Download Docker Desktop

# Start services
docker-compose up -d

# Verify services
docker-compose ps
# Should show: langfuse, mcp-gateway, price-api-mock
```

### **3. LangFuse (Observability)**

```bash
# Start LangFuse
docker-compose up -d langfuse

# Verify LangFuse
curl http://localhost:3000/health
# Should return: {"status": "healthy"}

# Access LangFuse UI
open http://localhost:3000
```

---

## ⚙️ **Configuration Setup**

### **1. Environment Variables**

```bash
# Create .env file
cp .env.example .env

# Edit .env file with your values
nano .env
```

**Required Environment Variables:**
```bash
# Organization settings
ORG_NAME="DemoOrg"
ENVIRONMENT="local"

# LLM Gateway
CENTRAL_LLM_GATEWAY_URL="http://localhost:11434"
DEFAULT_EMBEDDING_MODEL="all-MiniLM-L6-v2"

# MCP Gateway
CENTRAL_MCP_GATEWAY_URL="http://localhost:8081"

# LangFuse (Observability)
LANGFUSE_PUBLIC_KEY="your_public_key"
LANGFUSE_SECRET_KEY="your_secret_key"
LANGFUSE_HOST="http://localhost:3000"

# Optional: LangSmith (Alternative observability)
LANGSMITH_API_KEY="your_api_key"
```

### **2. Knowledge Base Setup**

```bash
# Create knowledge base directory
mkdir -p kb

# Add sample runbooks
cat > kb/basket_segments_runbook.md << 'EOF'
# Basket Segments File Drop Failure

## Issue Description
Basket segments feed from LHS (file drop process) fails with timeout errors.

## Symptoms
- Splunk alerts showing `CreateBasketSegmentsProcessor` errors
- Error message: `file pick-up process failed`
- Timeout: `java.io.InterruptedIOException: timeout`

## Diagnosis Steps
1. Check Splunk logs for timeout patterns
2. Verify file path accessibility
3. Check network connectivity
4. Review processor configuration

## Resolution
1. Restart the processor service
2. Clear stuck files from processing queue
3. Update timeout configuration
4. Monitor for recurring issues
EOF

# Add more runbooks as needed
```

### **3. Mock Data Setup**

```bash
# Create mock tickets directory
mkdir -p data/mock_tickets

# Add sample tickets
cat > data/mock_tickets/ALERT-001.json << 'EOF'
{
  "id": "ALERT-001",
  "subject": "Basket segments file drop failure",
  "description": "CreateBasketSegmentsProcessor is failing with timeout errors",
  "priority": "high",
  "status": "new",
  "created_at": "2024-01-15T10:30:00Z"
}
EOF

# Add more mock tickets as needed
```

---

## 🧪 **Verification Tests**

### **1. Basic Installation Test**

```bash
# Test core imports
python -c "
import langgraph
import langfuse
import faiss
import sentence_transformers
print('✅ Core dependencies imported successfully')
"

# Test framework imports
python -c "
from core.config import load_config
from agents.triage.agent import TriageAgent
from core.rag.local_kb import LocalKB
print('✅ Framework components imported successfully')
"
```

### **2. Service Connectivity Test**

```bash
# Test Ollama
curl http://localhost:11434/api/tags
# Should return: {"models": [{"name": "llama3.2", ...}]}

# Test MCP Gateway
curl http://localhost:8081/health
# Should return: {"status": "healthy"}

# Test LangFuse
curl http://localhost:3000/health
# Should return: {"status": "healthy"}
```

### **3. Framework Smoke Test**

```bash
# Run smoke test
python scripts/smoke_test.py
# Should complete without errors

# Run memory test
python scripts/test_memory_duplicate.py
# Should show FAISS memory working

# Run evaluation test
python scripts/test_evaluation_integration.py
# Should show evaluation components working
```

### **4. Full Workflow Test**

```bash
# Run end-to-end test
python scripts/test_full_workflow.py
# Should show complete agent workflow
```

---

## 🐛 **Troubleshooting**

### **Common Issues**

#### **1. Python Version Issues**
```bash
# Check Python version
python --version
# Should be 3.9+

# If using pyenv
pyenv install 3.11.0
pyenv local 3.11.0
```

#### **2. Dependency Conflicts**
```bash
# Clean install
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### **3. Ollama Issues**
```bash
# Restart Ollama
pkill ollama
ollama serve

# Reinstall model
ollama rm llama3.2
ollama pull llama3.2
```

#### **4. Docker Issues**
```bash
# Restart Docker
docker-compose down
docker-compose up -d

# Check logs
docker-compose logs langfuse
docker-compose logs mcp-gateway
```

#### **5. LangFuse Issues**
```bash
# Reset LangFuse
docker-compose down langfuse
docker-compose up -d langfuse

# Check database
docker-compose exec langfuse psql -U postgres -d langfuse
```

### **Performance Issues**

#### **1. Memory Issues**
```bash
# Check memory usage
free -h  # Linux
vm_stat  # macOS

# Reduce model size
ollama pull llama3.2:1b  # Use smaller model
```

#### **2. Slow Performance**
```bash
# Use CPU-optimized FAISS
pip uninstall faiss-cpu
pip install faiss-cpu --no-cache-dir

# Use smaller embedding model
export DEFAULT_EMBEDDING_MODEL="all-MiniLM-L6-v2"
```

---

## 🚀 **Quick Start Commands**

### **Complete Setup (5 minutes)**

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and setup
git clone https://github.com/your-org/golden-agent-framework.git
cd golden-agent-framework
uv venv && source .venv/bin/activate
uv install

# 3. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull llama3.2

# 4. Start services
docker-compose up -d

# 5. Setup environment
cp .env.example .env
# Edit .env with your values

# 6. Test installation
python scripts/smoke_test.py
```

### **Development Setup**

```bash
# Install development dependencies
uv install --dev

# Run tests
pytest

# Run linting
black .
flake8 .

# Run type checking
mypy .
```

---

## 📊 **Installation Verification**

### **Checklist**

- [ ] Python 3.9+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Ollama running with llama3.2
- [ ] Docker services up
- [ ] LangFuse accessible
- [ ] Environment variables set
- [ ] Knowledge base files present
- [ ] Mock tickets created
- [ ] Smoke test passes
- [ ] Full workflow test passes

### **Success Indicators**

```bash
# All services healthy
curl http://localhost:11434/api/tags | jq '.models[0].name'  # "llama3.2"
curl http://localhost:8081/health | jq '.status'            # "healthy"
curl http://localhost:3000/health | jq '.status'           # "healthy"

# Framework working
python scripts/smoke_test.py | grep "✅"                    # Multiple success messages
python scripts/test_full_workflow.py | grep "✅"            # Workflow completed
```

---

## 🎯 **Next Steps**

### **1. Run Demo**
```bash
# Quick demo
python scripts/test_full_workflow.py

# Open LangFuse
open http://localhost:3000/traces
```

### **2. Explore Framework**
```bash
# Check available agents
python -c "from agents import *; print([x for x in dir() if 'Agent' in x])"

# Check available tools
python -c "from core.gateway.tool_registry import ToolRegistry; print('Tools available')"

# Check RAG system
python -c "from core.rag.local_kb import LocalKB; kb = LocalKB('kb'); kb.load(); print(f'KB loaded: {len(kb._docs)} docs')"
```

### **3. Customize Configuration**
```bash
# Edit agent configuration
nano config/agent.yaml

# Add custom tools
nano config/gateway.yaml

# Add custom prompts
nano prompts/triage/system_prompt.md
```

---

## 📚 **Additional Resources**

### **Documentation**
- [Architecture Guide](docs/architecture.md)
- [Agent Patterns](docs/agents.md)
- [RAG System](docs/MEMORY_AGENT_FAISS.md)
- [Evaluation System](docs/evaluation_integration_guide.md)

### **Demo Scripts**
- [Demo Plan](AI_FORUM_DEMO_PLAN.md)
- [Execution Guide](DEMO_EXECUTION_GUIDE.md)
- [Quick Reference](DEMO_QUICK_REFERENCE.md)

### **Support**
- GitHub Issues: [Create Issue](https://github.com/your-org/golden-agent-framework/issues)
- Documentation: [Read the Docs](https://golden-agent-framework.readthedocs.io)
- Community: [Discord Server](https://discord.gg/golden-agent-framework)

---

**🎯 Installation complete! The Golden Agent Framework is ready for development and demo.**
