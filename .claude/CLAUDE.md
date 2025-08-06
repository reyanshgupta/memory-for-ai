# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development and Running
- **Local server (stdio)**: `uv run memory-server` - For local Claude Desktop
- **Remote server (HTTP)**: `uv run memory-server-http` - For remote Claude Desktop access
- **Easy remote startup**: `./start-remote-server.sh` - Startup script with configuration info
- **Install dependencies**: `uv sync`
- **Install UV**: Follow [UV installation guide](https://docs.astral.sh/uv/getting-started/installation/)

### Remote Setup Requirements
- **mcp-remote**: `npm install -g mcp-remote` (for Claude Desktop remote connection)
- **Node.js**: Required for mcp-remote proxy

### Configuration
- **Entry points**: 
  - `memory-server` → `app.main:main` (stdio transport)
  - `memory-server-http` → `app.http_server:main` (HTTP+SSE transport)
- **Environment prefix**: All config uses `MEMORY_SERVER_` prefix
- **Data directory**: `.memory-server/` (default, configurable via `MEMORY_SERVER_DATA_DIR`)

## Architecture Overview

### Core Components
1. **FastMCP Server** (`app/main.py`): Local MCP stdio transport implementation
2. **HTTP+SSE Server** (`app/http_server.py`): Remote HTTP transport with FastAPI
3. **Vector Store** (`app/storage/vector_store.py`): ChromaDB persistence with semantic search
4. **Embedding Service** (`app/search/embeddings.py`): sentence-transformers for local embeddings
5. **Memory Models** (`app/models/`): Pydantic v2 models for data validation
6. **Configuration** (`app/models/config.py`): Environment-based settings with validation

### Data Flow
- Memory content → Embedding service → Vector embedding → ChromaDB storage
- Search queries → Embedding → Vector similarity search → Ranked results
- All processing is local (no external API calls)

### MCP Tools Available
- `health_check`: Server status and statistics
- `add_memory`: Store content with automatic embedding generation
- `search_memories`: Semantic search with similarity scoring
- `get_memory`: Retrieve by ID
- `delete_memory`: Remove from storage

### Key Design Patterns
- **Global app context**: `_app_context` provides service access across MCP tools
- **Async/await throughout**: All operations are async for MCP compatibility
- **Lazy loading**: Embedding model loads on first use
- **Error handling**: Try/catch with logging and graceful error responses
- **Enum validation**: Memory types/sources are validated against predefined enums

### Technology Stack
- **MCP Protocol**: FastMCP (stdio) + FastAPI (HTTP+SSE) for Claude Desktop integration
- **HTTP Server**: FastAPI with CORS for remote access
- **Vector DB**: ChromaDB with disabled telemetry for local storage
- **Embeddings**: all-MiniLM-L6-v2 via sentence-transformers
- **Config**: Pydantic Settings with environment variable support
- **Package Management**: UV for dependency management and execution

### Memory Model Structure
```python
Memory {
  id: str (UUID)
  content: str
  type: MemoryType (text|image|document|conversation|code)
  source: MemorySource (claude|chatgpt|cursor|browser|api|manual)  
  tags: List[str]
  metadata: Dict[str, Any]
  embedding: List[float] (384 dimensions)
  importance: float (0.0-10.0)
  created_at/updated_at: datetime
}
```

### Environment Configuration
All settings use `MEMORY_SERVER_` prefix:
- `MEMORY_SERVER_DATA_DIR`: Storage location
- `MEMORY_SERVER_EMBEDDING_MODEL_NAME`: Transformer model
- `MEMORY_SERVER_LOG_LEVEL`: Logging verbosity
- See `app/models/config.py` for complete list

### Client Integration

#### Local Integration (stdio transport)
Configure Claude Desktop with `examples/claude_desktop_config.json` for local server.

#### Remote Integration (HTTP+SSE transport)
1. **Start remote server**: `./start-remote-server.sh` or `uv run memory-server-http`
2. **Install mcp-remote**: `npm install -g mcp-remote`
3. **Configure Claude Desktop** with `examples/remote_claude_desktop_config.json`:
   - Location: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
   - Location: `%APPDATA%\Claude\claude_desktop_config.json` (Windows)
   - Use mcp-remote proxy to bridge stdio ↔ HTTP+SSE transport
4. **Restart Claude Desktop** after config changes

#### Remote Server Configuration
- **Host**: `MEMORY_SERVER_HOST=localhost` (default)
- **Port**: `MEMORY_SERVER_PORT=8000` (default)
- **URL**: Server accessible at `http://localhost:8000/mcp`
- **Health Check**: `http://localhost:8000/` returns server info