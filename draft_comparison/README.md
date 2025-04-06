# Chain of Draft Comparison

A web-based interface for comparing different Chain of Draft approaches:
- Batch mode: Generate all drafts at once
- Sequential mode: Generate each draft separately
- Mirror mode: Use reflective questions between drafts

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python run_comparison_server.py
```

This will start both:
- HTTP server for the frontend (http://localhost:8000)
- WebSocket server for real-time communication (ws://localhost:765)

3. Open in browser:
```
http://localhost:8000/frontend/comparison.html
```

## Project Structure

```
draft_comparison/
├── frontend/
│   └── comparison.html    # Web interface
├── backend/
│   ├── __init__.py
│   └── comparison_server.py   # WebSocket server
├── examples/
│   └── chain_of_draft_comparison.py  # Core comparison logic
├── requirements.txt
├── README.md
└── run_comparison_server.py   # Main entry point
```

## Features

- Real-time progress tracking
- Side-by-side comparison of approaches
- Token usage and cost metrics
- Draft-by-draft comparison
- Clean server shutdown from UI

## Usage

1. Enter your problem in the text area
2. Select the model to use
3. Click "Run Comparison"
4. View results in three tabs:
   - Metrics & Costs
   - Draft Comparison
   - Final Answers
5. Use "Stop Server" to cleanly shut down when done

## Development

The project uses:
- Vanilla JavaScript for frontend
- WebSocket for real-time communication
- Python's asyncio for the server
- Chain of Draft implementation from PromptChain 