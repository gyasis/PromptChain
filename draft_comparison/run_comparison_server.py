import asyncio
import multiprocessing
import http.server
import socketserver
import os
import signal
import sys
import socket

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from draft_comparison.backend.comparison_server import ComparisonServer

def find_available_port(start_port, max_attempts=100):
    """Find the next available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

def run_http_server(port=8000):
    """Run the HTTP server for the frontend."""
    # Find available port starting from the specified port
    available_port = find_available_port(port)
    
    # Change to the project root directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", available_port), handler) as httpd:
        print(f"HTTP server running at http://localhost:{available_port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("HTTP server stopping...")
            httpd.shutdown()

def run_websocket_server(start_port=8765):
    """Run the WebSocket server."""
    server = ComparisonServer(start_port=start_port)
    asyncio.run(server.start())
    return server.port

def main():
    # Start HTTP server in a separate process
    http_port = find_available_port(8000)
    http_process = multiprocessing.Process(
        target=run_http_server,
        args=(http_port,)
    )
    http_process.start()

    # Start WebSocket server in a separate process
    ws_port = find_available_port(8765)
    ws_process = multiprocessing.Process(
        target=run_websocket_server,
        args=(ws_port,)
    )
    ws_process.start()

    print("\nServers started!")
    print(f"Open http://localhost:{http_port}/frontend/comparison.html?ws_port={ws_port} in your browser")
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