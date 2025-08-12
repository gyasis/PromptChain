#!/usr/bin/env python3
"""
Start the Research Agent FastAPI backend server
"""

import uvicorn
import sys
import os
from pathlib import Path

# Add the backend to the path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

if __name__ == "__main__":
    print("🔬 Starting Research Agent Backend...")
    print("📚 API Documentation: http://localhost:8078/api/docs")
    print("🔄 WebSocket Progress: ws://localhost:8078/ws/progress/{session_id}")
    print("💬 WebSocket Chat: ws://localhost:8078/ws/chat/{session_id}")
    print("🛑 Press Ctrl+C to stop\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8078,
        reload=True,
        log_level="info",
        reload_dirs=[str(backend_dir)]
    )