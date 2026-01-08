#!/usr/bin/env python3
"""Simple HTTP health check endpoint for monitoring."""

import json
import subprocess
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from pathlib import Path


class HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check endpoint."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/health':
            # Check if bots are running
            bots_running = self.check_bots()
            
            # Check if state files exist (indicates recent activity)
            state_files = self.check_state_files()
            
            status = "healthy" if bots_running > 0 else "unhealthy"
            
            response = {
                "status": status,
                "bots_running": bots_running,
                "state_files_count": state_files,
                "timestamp": datetime.now().isoformat()
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response, indent=2).encode())
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b"""
            <html>
            <head><title>Trading Bot Health Check</title></head>
            <body>
                <h1>Trading Bot System Health</h1>
                <p><a href="/health">Health Status (JSON)</a></p>
            </body>
            </html>
            """)
        else:
            self.send_response(404)
            self.end_headers()
    
    def check_bots(self):
        """Check if bots are running by counting processes."""
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'python.*_bot.py'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                return len([p for p in result.stdout.strip().split('\n') if p])
            return 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return 0
    
    def check_state_files(self):
        """Count state files to indicate recent activity."""
        try:
            state_files = list(Path('.').rglob('*_state.json'))
            return len(state_files)
        except Exception:
            return 0
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def main():
    """Run the health check server."""
    port = 8080
    
    # Allow port override via environment variable
    import os
    if os.getenv('HEALTH_CHECK_PORT'):
        try:
            port = int(os.getenv('HEALTH_CHECK_PORT'))
        except ValueError:
            print(f"Invalid port: {os.getenv('HEALTH_CHECK_PORT')}, using default 8080")
    
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    print(f"Health check server running on http://0.0.0.0:{port}/health")
    print("Press Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down health check server...")
        server.shutdown()
        sys.exit(0)


if __name__ == '__main__':
    main()
