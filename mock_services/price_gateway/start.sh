#!/bin/bash
# Start script for Price Gateway (with internal backend)

set -e

echo "🚀 Starting Price Gateway..."

# Start backend API (internal, port 8090)
echo "   → Starting backend API on port 8090..."
python backend/app.py &
BACKEND_PID=$!

# Wait for backend to be ready
sleep 2

# Start gateway (external, port 8082)
echo "   → Starting gateway on port 8082..."
python app.py &
GATEWAY_PID=$!

echo "✅ Price Gateway started!"
echo "   Backend API: http://127.0.0.1:8090 (internal)"
echo "   Gateway:     http://0.0.0.0:8082 (external)"

# Wait for both processes
wait $BACKEND_PID $GATEWAY_PID

