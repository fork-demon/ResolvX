## Price Lifecycle Mock Service (FastAPI + Docker)

A dockerized mock of price lifecycle endpoints for demos/CI. Mirrors the curls you provided.

### Endpoints
- POST `/pricelifecycle/v1/minimumPrices/calculate`
- GET `/pricelifecycle/v1/minimumPrices`
- GET `/pricelifecycle/v1/basketSegments`
- GET `/pricelifecycle/v1/competitorPrices`
- GET `/pricelifecycle/v1/advisory/promotions/competitor-promotional-prices`
- GET `/pricelifecycle/v1/advisory/promotions/effectiveness`
- POST `/pricelifecycle/v1/policies/view`
- GET `/pricelifecycle/v5/basePrices`

### Run with Docker
```bash
docker build -t price-api-mock ./mock_services/price_api
docker run --rm -p 8090:8080 --name price-api-mock price-api-mock
```
Service is available at `http://localhost:8090`.

### Example Calls
```bash
curl --request POST \
  --url http://localhost:8090/pricelifecycle/v1/minimumPrices/calculate \
  --header 'accept: application/json' \
  --header 'authorization: Bearer xyz' \
  --header 'content-type: application/json' \
  --header 'teamnumber: 100' \
  --header 'traceid: CalculateMinPrice:1234' \
  --data '{
  "gtin": "05013967015241",
  "locationClusterIds": [
    "f6458c43-25c0-4c24-935f-4f284041d573",
    "212a9988-a503-4f30-bacd-a9d672ec2c9a"
  ],
  "qtyContents": {"totalQuantity": 100, "quantityUom": "cl"},
  "taxTypeCode": "411",
  "supplierABV": "40.0"
}'
```

### Optional: docker-compose (with MCP gateway)
```yaml
services:
  price-api-mock:
    build: ./mock_services/price_api
    ports:
      - "8090:8080"
  mock-mcp-gateway:
    build: ./mock_services/mcp_gateway
    ports:
      - "8081:8081"
    environment:
      - PRICE_API_BASE_URL=http://price-api-mock:8080
    depends_on:
      - price-api-mock
```

### Integrate with Agents
- Preferred: use the `mock-mcp-gateway` and map tools as `type: mcp` with `server: mock-mcp-gateway` in `gateway.tools`.
- Quick dev: add a local tool function that calls `http://localhost:8090/...` and map as `type: local`.


