# Enhanced Chatbot System

![CI Status](https://github.com/akabarki76/CHATBOTSYS/workflows/CI/badge.svg)

A production-ready chatbot system with:
- Multi-turn conversation memory
- Command handling system
- Plugin architecture
- External API integration
- FastAPI web interface

## Features

- **Conversation Management**: Maintains context with automatic pruning
- **Command System**: Built-in commands (`/weather`, `/time`, `/help`) + plugins
- **Secure Configuration**: Environment variables + config file
- **Production Ready**: Docker support, health checks, logging

## Quick Start

```bash
# Clone repository
git clone https://github.com/akabarki76/CHATBOTSYS
cd CHATBOTSYS

# Create config (copy and edit example)
cp config.example.json config.json

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn src.main:app --reload
