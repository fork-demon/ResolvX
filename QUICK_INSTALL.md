# ⚡ Quick Install - Golden Agent Framework

## 🚀 **One-Line Installation**

### **macOS/Linux:**
```bash
curl -fsSL https://raw.githubusercontent.com/your-org/golden-agent-framework/main/setup.sh | bash
```

### **Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/your-org/golden-agent-framework/main/setup.ps1" -OutFile "setup.ps1"; .\setup.ps1
```

---

## 🛠️ **Manual Installation (5 minutes)**

### **1. Clone Repository**
```bash
git clone https://github.com/your-org/golden-agent-framework.git
cd golden-agent-framework
```

### **2. Run Setup Script**
```bash
# macOS/Linux
./setup.sh

# Windows
.\setup.ps1
```

### **3. Verify Installation**
```bash
# Test the framework
python scripts/smoke_test.py

# Run full demo
python scripts/test_full_workflow.py
```

---

## 📋 **Prerequisites**

- **Python 3.9+**
- **Docker Desktop** (for services)
- **Git** (for cloning)

### **Optional:**
- **Ollama** (for local LLM)
- **uv** (for faster package management)

---

## 🎯 **What Gets Installed**

- ✅ **Python Dependencies**: All required packages
- ✅ **Ollama LLM**: llama3.2 model
- ✅ **Docker Services**: LangFuse, MCP Gateway, Price API
- ✅ **Configuration**: Environment variables and settings
- ✅ **Mock Data**: Sample tickets and knowledge base
- ✅ **Verification**: Automated tests

---

## 🚨 **Troubleshooting**

### **If setup fails:**
```bash
# Check Python version
python --version  # Should be 3.9+

# Check Docker
docker --version

# Check services
docker-compose ps
```

### **If tests fail:**
```bash
# Restart services
docker-compose down && docker-compose up -d

# Reinstall dependencies
uv install --force
```

---

## 📚 **Next Steps**

1. **Edit Configuration**: Update `.env` file
2. **Add Knowledge Base**: Add runbooks to `kb/` directory
3. **Run Demo**: `python scripts/test_full_workflow.py`
4. **View Traces**: Open http://localhost:3000

---

**🎯 Ready to go! The framework will be fully functional after installation.**
