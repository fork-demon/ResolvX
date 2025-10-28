# Price Team Gateway

**Self-contained MCP gateway with internal backend API**

## 📦 Structure

```
price_gateway/
├── app.py                    # Gateway HTTP wrapper (port 8082)
├── server.py                 # MCP SDK server
├── backend/
│   ├── app.py               # Internal Price API (port 8090)
│   ├── Dockerfile           # (not used, integrated into parent)
│   └── requirements.txt     # (not used, integrated into parent)
├── Dockerfile               # Builds both gateway + backend
├── start.sh                 # Local startup script
└── docker-entrypoint.sh     # Docker startup script
```

## 🔧 Architecture

```
┌────────────────────────────────────────┐
│      Price Team Gateway (8082)         │
│  ┌──────────────────────────────────┐  │
│  │   Gateway Layer (MCP)            │  │
│  │   - base_prices_get              │  │
│  │   - competitor_prices_get        │  │
│  │   - product_info_get             │  │
│  │   - location_info_get            │  │
│  └──────────┬───────────────────────┘  │
│             │                           │
│             │ HTTP (internal)           │
│             ▼                           │
│  ┌──────────────────────────────────┐  │
│  │   Backend API (8090)             │  │
│  │   - /pricelifecycle/v5/basePrices│  │
│  │   - /pricelifecycle/v1/competitor│  │
│  │   - /pricelifecycle/v1/basket... │  │
│  │   - /pricelifecycle/v1/policies  │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
         External access: 8082 only
         Internal: 8090 (localhost only)
```

## 🚀 Running Locally

### Start both services:
```bash
cd mock_services/price_gateway
./start.sh
```

Or manually:
```bash
# Terminal 1: Backend API
python backend/app.py

# Terminal 2: Gateway
python app.py
```

### Test:
```bash
# Test gateway
curl http://localhost:8082/health

# Test tool execution
curl -X POST http://localhost:8082/mcp \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "tools/execute",
    "params": {
      "tool_name": "base_prices_get",
      "parameters": {
        "tpnb": "12345678",
        "locationClusterId": "LC001"
      }
    }
  }'
```

## 🐳 Running with Docker

```bash
# Build
docker build -t price-gateway .

# Run
docker run -p 8082:8082 price-gateway

# Or with docker-compose
docker-compose up price-gateway
```

## 🔧 Configuration

### Environment Variables

- `PRICE_API_BASE_URL`: Backend API URL (default: `http://127.0.0.1:8090`)
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `DEBUG`: Enable debug mode for error details

### Ports

- **8082**: Gateway (external, MCP JSON-RPC)
- **8090**: Backend API (internal, REST)

## 📝 Tools Provided

### 1. `base_prices_get`
Get base price for a product at a location cluster.
- **Calls**: `GET /pricelifecycle/v5/basePrices`
- **Fallback**: Mock data if backend fails

### 2. `competitor_prices_get`
Get competitor pricing data for a product.
- **Calls**: `GET /pricelifecycle/v1/competitorPrices`
- **Fallback**: Mock data if backend fails

### 3. `product_info_get`
Get detailed product information.
- **Source**: Pure mock data (not from backend)

### 4. `location_info_get`
Get location/store information.
- **Source**: Pure mock data (not from backend)

## 🎯 Benefits

✅ **Self-contained**: No external dependencies  
✅ **Modular**: Gateway + Backend in one package  
✅ **Realistic**: Calls actual REST API internally  
✅ **Fallback**: Graceful degradation if backend fails  
✅ **Secure**: Backend only accessible via localhost  
✅ **Easy Deploy**: Single Docker container  

## 🔍 Testing Backend Directly

```bash
# Get base prices
curl "http://127.0.0.1:8090/pricelifecycle/v5/basePrices?tpnb=12345678&locationClusterId=LC001"

# Calculate minimum price
curl -X POST http://127.0.0.1:8090/pricelifecycle/v1/minimumPrices/calculate \
  -H 'Content-Type: application/json' \
  -d '{
    "gtin": "5000112345678",
    "locationClusterIds": ["LC001"],
    "qtyContents": {
      "totalQuantity": 2.0,
      "quantityUom": "cl"
    },
    "taxTypeCode": "VAT",
    "supplierABV": "5.0"
  }'
```

## 📊 Monitoring

### Health Checks
```bash
# Gateway health
curl http://localhost:8082/health

# Backend health (from within container/localhost only)
curl http://127.0.0.1:8090/health
```

### Logs
- Gateway logs: stdout (FastAPI/Uvicorn)
- Backend logs: stdout (FastAPI/Uvicorn)
- Both are captured in Docker logs: `docker logs <container_id>`

