# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the LiveKit Agents framework, a Python-based framework for building realtime voice AI agents. The codebase contains:

- **Core framework** (`livekit-agents/`): Main agents library with voice, LLM, STT, TTS, and orchestration capabilities
- **Plugin ecosystem** (`livekit-plugins/`): Provider-specific plugins for various AI services (OpenAI, Anthropic, Deepgram, etc.)
- **Examples** (`examples/`): Comprehensive examples for different use cases
- **Tests** (`tests/`): Test suite with Docker-based infrastructure

## Development Commands

### Package Management
- `uv sync --all-extras --dev` - Install all dependencies including dev extras
- `uv run <command>` - Run commands in the UV environment

### Code Quality
- `ruff check --fix` - Run linting with auto-fixes
- `ruff format` - Format code according to project style
- `mypy` - Run type checking

### Testing
- `pytest` - Run test suite
- `pytest -s --color=yes --tb=short tests/test_<module>.py` - Run specific test module
- `make -C tests test PLUGIN=<plugin_name>` - Run Docker-based plugin tests

### Running Agents
- `python agent.py console` - Run agent in terminal mode for local testing
- `python agent.py dev` - Run in development mode with hot reloading
- `python agent.py start` - Run in production mode

## Architecture Overview

### Core Components

#### Agent System
- **Agent** (`livekit/agents/voice/agent.py`): Main agent class with instructions, tools, and model configuration
- **AgentSession** (`livekit/agents/voice/agent_session.py`): Manages agent lifecycle and user interactions
- **Worker** (`livekit/agents/worker.py`): Main process that coordinates job scheduling and launches agents

#### Voice Pipeline
- **STT** (`livekit/agents/stt/`): Speech-to-text abstraction and implementations
- **LLM** (`livekit/agents/llm/`): Language model abstraction with chat context, tools, and realtime support
- **TTS** (`livekit/agents/tts/`): Text-to-speech abstraction and implementations
- **VAD** (`livekit/agents/vad.py`): Voice activity detection

#### Plugin Architecture
Each plugin in `livekit-plugins/` follows a consistent structure:
- Provider-specific implementations of STT, LLM, or TTS interfaces
- Models and configuration classes
- Version and logging utilities

### Key Design Patterns

#### Entry Points
Agents use an `entrypoint` function similar to web request handlers:
```python
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    # Agent setup and session management
```

#### Tool System
Function tools are created using the `@function_tool` decorator and automatically discovered from agent classes.

#### Context Management
- `JobContext`: Per-job execution context
- `RunContext`: Function tool execution context
- `ChatContext`: LLM conversation state management

## Development Guidelines

### Code Style
- Line length: 100 characters
- Target Python 3.9+
- Use ruff for linting and formatting
- Follow Google docstring conventions

### Plugin Development
When creating new plugins:
1. Follow existing plugin structure in `livekit-plugins/`
2. Implement appropriate base classes (STT, LLM, TTS)
3. Include proper error handling and logging
4. Add version.py and models.py files
5. Update pyproject.toml with dependencies

### Testing
- Use pytest with asyncio support
- Docker-based testing infrastructure available
- Test files include fake implementations for mocking

### Environment Variables
Common environment variables for development:
- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` - LiveKit server connection
- Provider-specific API keys (e.g., `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`)

### MCP Support
The framework has native MCP (Model Context Protocol) support for integrating external tools via MCP servers.