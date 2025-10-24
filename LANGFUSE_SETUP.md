# LangFuse Setup Guide

## Quick Setup for Local Development

### Option 1: Use LangFuse Cloud (Easiest)
If you want to use the cloud version of LangFuse:

1. **Sign up at [LangFuse Cloud](https://cloud.langfuse.com)**
2. **Get your API keys** from the dashboard
3. **Set environment variables:**
   ```bash
   export LANGFUSE_PUBLIC_KEY="your-public-key"
   export LANGFUSE_SECRET_KEY="your-secret-key"
   export LANGFUSE_HOST="https://cloud.langfuse.com"
   ```

### Option 2: Run LangFuse Locally (Self-hosted)

#### Step 1: Start LangFuse Services
```bash
# Start LangFuse with Docker Compose
docker-compose -f docker-compose.langfuse.yml up -d

# Check if services are running
docker-compose -f docker-compose.langfuse.yml ps
```

#### Step 2: Access LangFuse UI
- Open http://localhost:3000 in your browser
- Create an account (first user becomes admin)
- Go to Settings → API Keys
- Copy your Public Key and Secret Key

#### Step 3: Configure Environment Variables
```bash
export LANGFUSE_PUBLIC_KEY="your-public-key"
export LANGFUSE_SECRET_KEY="your-secret-key"
export LANGFUSE_HOST="http://localhost:3000"
```

### Step 4: Test the Setup

#### Test 1: Check LangFuse Connection
```bash
python -c "
from core.observability.langfuse_tracer import LangFuseTracer
import os

# Set environment variables if not already set
os.environ.setdefault('LANGFUSE_PUBLIC_KEY', 'your-public-key')
os.environ.setdefault('LANGFUSE_SECRET_KEY', 'your-secret-key')
os.environ.setdefault('LANGFUSE_HOST', 'http://localhost:3000')

tracer = LangFuseTracer()
print('✅ LangFuse tracer initialized successfully')
"
```

#### Test 2: Run the Full Workflow
```bash
python scripts/test_full_workflow.py
```

#### Test 3: Check Traces in LangFuse UI
1. Go to http://localhost:3000 (or your LangFuse URL)
2. Navigate to "Traces" section
3. You should see traces from the workflow

## Troubleshooting

### Issue 1: "No traces appearing"
**Solutions:**
1. Check if LangFuse is running: `curl http://localhost:3000/api/public/health`
2. Verify API keys are correct
3. Check network connectivity
4. Look at application logs for errors

### Issue 2: "Connection refused"
**Solutions:**
1. Make sure LangFuse services are running
2. Check if ports are available (3000, 5433, 8124)
3. Try different ports if conflicts exist

### Issue 3: "Authentication failed"
**Solutions:**
1. Verify API keys are correct
2. Check if the project exists in LangFuse
3. Ensure the host URL is correct

### Issue 4: "Traces not showing input/output"
**Solutions:**
1. Check if spans are being created with proper data
2. Verify the tracer is properly initialized
3. Look for errors in the application logs

## Environment Variables Reference

```bash
# Required
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
LANGFUSE_HOST=http://localhost:3000  # or https://cloud.langfuse.com

# Optional
LANGFUSE_PROJECT_NAME=golden-agents
LANGFUSE_ENVIRONMENT=development
```

## Quick Commands

```bash
# Start LangFuse
docker-compose -f docker-compose.langfuse.yml up -d

# Stop LangFuse
docker-compose -f docker-compose.langfuse.yml down

# View logs
docker-compose -f docker-compose.langfuse.yml logs -f

# Reset everything
docker-compose -f docker-compose.langfuse.yml down -v
docker-compose -f docker-compose.langfuse.yml up -d
```
