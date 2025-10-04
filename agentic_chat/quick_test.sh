#!/bin/bash
# Quick test of agentic chat with Candle library query

echo "can you give me a cliff notes version including examples of using Rust's Candle. After, can you also give me alternatives to python's litellm library in rust?" | python agentic_team_chat.py --quiet --no-logging
