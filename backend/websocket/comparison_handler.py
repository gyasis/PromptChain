from typing import Dict, Any
from websockets.server import WebSocketServerProtocol
from ..services.comparison_service import ComparisonService

class ComparisonHandler:
    def __init__(self):
        self.comparison_service = ComparisonService()

    async def handle_message(self, websocket: WebSocketServerProtocol, message: Dict[str, Any]):
        """Handle incoming WebSocket messages."""
        try:
            if message['type'] == 'run_comparison':
                await self._handle_comparison(websocket, message['payload'])
            else:
                await websocket.send({
                    'type': 'error',
                    'payload': {'message': 'Unknown message type'}
                })
        except Exception as e:
            await websocket.send({
                'type': 'error',
                'payload': {'message': str(e)}
            })

    async def _handle_comparison(self, websocket: WebSocketServerProtocol, params: Dict[str, Any]):
        """Handle a comparison request."""
        # Send initial status
        await websocket.send({
            'type': 'status',
            'payload': {'status': 'starting'}
        })

        # Set up progress callback
        async def progress_callback(mode: str, progress: float):
            await websocket.send({
                'type': 'progress',
                'payload': {
                    'mode': mode,
                    'progress': round(progress * 100)
                }
            })

        # Run comparison
        try:
            results = await self.comparison_service.run_comparison(
                problem=params['problem'],
                model=params['model'],
                progress_callback=progress_callback
            )

            # Send final results
            await websocket.send({
                'type': 'results',
                'payload': results
            })

        except Exception as e:
            await websocket.send({
                'type': 'error',
                'payload': {'message': str(e)}
            })

    async def handle_connect(self, websocket: WebSocketServerProtocol):
        """Handle new WebSocket connection."""
        await websocket.send({
            'type': 'status',
            'payload': {'status': 'connected'}
        })

    async def handle_disconnect(self, websocket: WebSocketServerProtocol):
        """Handle WebSocket disconnection."""
        # Cleanup any resources if needed
        pass 