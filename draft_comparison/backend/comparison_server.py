import asyncio
import websockets
import json
import signal
import sys
import os
import logging
from ..examples.chain_of_draft_comparison import ChainOfDraftComparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

class ComparisonServer:
    def __init__(self):
        self.comparison = ChainOfDraftComparison()
        self.stop_event = asyncio.Event()

    async def handle_message(self, websocket):
        """Handle incoming WebSocket messages."""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    action = data.get('action')

                    if action == 'compare':
                        problem = data.get('problem')
                        if not problem:
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'payload': 'No problem provided'
                            }))
                            continue

                        # Run comparison
                        try:
                            await websocket.send(json.dumps({
                                'type': 'status',
                                'payload': 'Running comparison...'
                            }))

                            # Run in executor to not block event loop
                            comparison_result = await asyncio.get_event_loop().run_in_executor(
                                None, self.comparison.run_comparison, problem
                            )

                            # Send the result as direct JSON, not as a string
                            await websocket.send(json.dumps({
                                'type': 'comparison_result',
                                'payload': comparison_result
                            }))

                        except Exception as e:
                            logger.error(f"Error running comparison: {str(e)}", exc_info=True)
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'payload': f'Error running comparison: {str(e)}'
                            }))

                    elif action == 'stop':
                        logger.info("Stop command received")
                        self.stop_event.set()
                        await websocket.send(json.dumps({
                            'type': 'status',
                            'payload': 'Server stopping...'
                        }))
                        break

                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'payload': 'Invalid JSON message'
                    }))
                except Exception as e:
                    logger.error(f"Error handling message: {str(e)}", exc_info=True)
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'payload': f'Server error: {str(e)}'
                    }))

        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")

    async def start(self):
        """Start the WebSocket server."""
        async with websockets.serve(self.handle_message, "localhost", 8765):
            logger.info("server listening on 127.0.0.1:8765")
            print("Comparison server running at ws://localhost:8765")
            await self.stop_event.wait()  # Wait until stop event is set

if __name__ == '__main__':
    server = ComparisonServer()
    asyncio.run(server.start()) 