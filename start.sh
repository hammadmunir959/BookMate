#!/bin/bash

# RAG Microservice Startup Script
# This script starts both the backend microservice and frontend UI

echo "🚀 Starting RAG Microservice with UI..."

# Function to cleanup background processes
cleanup() {
    echo "🛑 Shutting down services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Trap SIGINT (Ctrl+C) to cleanup
trap cleanup SIGINT

# Activate Virtual Environment
if [ -d "venv" ]; then
    echo "🐍 Activating local venv..."
    source venv/bin/activate
fi

# Check if Python and Node.js are available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is required but not installed."
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ npm is required but not installed."
    exit 1
fi

# Start backend in background
echo "📡 Starting backend microservice..."
cd "$(dirname "$0")"
python3 main.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start"
    exit 1
fi

echo "✅ Backend started (PID: $BACKEND_PID)"

# Start frontend in background
echo "🖥️  Starting frontend UI..."
cd ui/
npm run dev &
FRONTEND_PID=$!

# Wait a moment for frontend to start
sleep 5

echo "✅ Frontend started (PID: $FRONTEND_PID)"
echo ""
echo "🌐 Services are running:"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:5173"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for background processes
wait
