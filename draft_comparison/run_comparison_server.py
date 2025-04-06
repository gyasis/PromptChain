import asyncio
import multiprocessing
import http.server
import socketserver
import os
import signal
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from draft_comparison.backend.comparison_server import ComparisonServer

def run_http_server(port=8000):
    """Run the HTTP server for the frontend."""
    # Change to the project root directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"HTTP server running at http://localhost:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("HTTP server stopping...")
            httpd.shutdown()

def run_websocket_server():
    """Run the WebSocket server."""
    server = ComparisonServer()
    asyncio.run(server.start())

def main():
    # Start HTTP server in a separate process
    http_process = multiprocessing.Process(
        target=run_http_server,
        args=(8000,)
    )
    http_process.start()

    # Start WebSocket server in a separate process
    ws_process = multiprocessing.Process(
        target=run_websocket_server
    )
    ws_process.start()

    print("\nServers started!")
    print("Open http://localhost:8000/frontend/comparison.html in your browser")
    print("Press Ctrl+C to stop all servers\n")

    try:
        # Wait for processes to complete
        http_process.join()
        ws_process.join()
    except KeyboardInterrupt:
        print("\nStopping servers...")
        # Send termination signal to processes
        http_process.terminate()
        ws_process.terminate()
        # Wait for processes to finish
        http_process.join()
        ws_process.join()
        print("Servers stopped!")

if __name__ == "__main__":
    main() 