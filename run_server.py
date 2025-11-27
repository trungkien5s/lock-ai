#!/usr/bin/env python3
"""
run_server.py - Start script cho Smart Locker API
Hỗ trợ cả local (SSL) và production (Render)
"""

import os
import sys
import ssl

if __name__ == "__main__":
    # Detect môi trường
    is_production = os.environ.get("RENDER") or os.environ.get("PRODUCTION")
    
    if is_production:
        # PRODUCTION: Render tự handle SSL qua reverse proxy
        port = int(os.environ.get("PORT", 8000))
        print(f"🚀 Starting in PRODUCTION mode on port {port}")
        
        import uvicorn
        uvicorn.run(
            "backend.main:app",
            host="0.0.0.0",
            port=port,
            log_level="info",
            reload=False
        )
    else:
        # LOCAL DEVELOPMENT: Dùng SSL
        print("🔒 Starting in LOCAL mode with SSL on https://localhost:8001")
        
        import uvicorn
        uvicorn.run(
            "backend.main:app",
            host="localhost",
            port=8001,
            ssl_keyfile='./ssl/privkey.pem',
            ssl_certfile='./ssl/fullchain.pem',
            log_level="info",
            reload=True  # Auto-reload cho development
        )