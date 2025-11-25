# Memory Server for AI Agents

A local MCP-compatible server that provides centralized memory storage for multiple AI tools including Claude Desktop, ChatGPT, browser agents, and other MCP-compatible clients.

## 🚀 Features

- 🧠 **Semantic Memory Storage**: Store and retrieve memories using vector embeddings
- 🔍 **Intelligent Search**: Search memories by meaning, not just keywords
- 🔌 **MCP Compatible**: Full Model Context Protocol (MCP) v2025-06-18 support
- 🏠 **Local-First**: No cloud dependencies, all data stays on your machine
- ⚡ **Fast & Efficient**: Built with UV package manager and optimized for performance
- 🔧 **Easy Integration**: Works seamlessly with Claude Desktop and other MCP clients

## 📦 Technology Stack

- **Protocol**: MCP (Model Context Protocol) with stdio/HTTP transports
- **Vector Database**: ChromaDB for semantic search and embedding storage
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2) for local text encoding
- **Configuration**: Pydantic v2 with environment variable support
- **Package Management**: UV for fast dependency management

## 🛠️ Installation

### Prerequisites

- Python 3.12+
- [UV package manager](https://docs.astral.sh/uv/getting-started/installation/)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/memory-for-ai.git
cd memory-for-ai

# Install dependencies with UV
uv sync

# The memory-server command is now available
uv run memory-server
```

### Automated Setup Scripts

For quick setup with Claude Desktop or Claude Code, use the provided setup scripts:

**Claude Desktop (macOS):**
```bash
./setup/claude-desktop/setup-mcp-macos.sh
```

**Claude Code (macOS):**
```bash
./setup/claude-code/setup-mcp-macos.sh
```

These scripts will:
- Detect your UV installation
- Install dependencies automatically
- Create or update the appropriate config file
- Back up any existing configuration

## 🎯 Quick Start

### 1. Start the Memory Server

```bash
uv run memory-server
```

You should see output like:
```
🧠 Starting Memory Server...
📋 Server Information:
   - Name: Memory Server
   - Version: 0.1.0
   - Protocol: MCP (Model Context Protocol)
   - Transport: stdio (Claude Desktop compatible)

🔧 Available Tools:
   - health_check - Check server status
   - add_memory - Store new memory with embedding
   - search_memories - Semantic search across memories
   - get_memory - Retrieve specific memory by ID
   - delete_memory - Remove memory from storage

✅ Memory Server ready for MCP connections!
```

### 2. Configure Claude Desktop

Add this configuration to your Claude Desktop config file:

**Location**: 
- macOS: `~/Library/Application Support/Claude/config.json`
- Windows: `%APPDATA%\Roaming\Claude\claude_desktop_config.json`

**Configuration**:
```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["run", "memory-server"],
      "cwd": "/path/to/your/memory-for-ai",
      "env": {
        "MEMORY_SERVER_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### 3. Start Using Memory

Once configured, you can use these commands in Claude Desktop:

- **Store a memory**: "Remember that I prefer TypeScript over JavaScript"
- **Search memories**: "What do you remember about my programming preferences?"
- **Get server status**: Ask Claude to check the memory server health

## 🔧 Available MCP Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `add_memory` | Store new memory with embedding | `content`, `memory_type`, `source`, `tags`, `importance`, `metadata` |
| `search_memories` | Semantic search across memories | `query`, `limit`, `filter_tags` |
| `get_memory` | Retrieve specific memory by ID | `memory_id` |
| `delete_memory` | Remove memory from storage | `memory_id` |
| `health_check` | Check server status and statistics | None |

## ⚙️ Configuration

Configuration is handled through environment variables. Create a `.env` file or set these variables:

```bash
# Storage Configuration
MEMORY_SERVER_DATA_DIR=.memory-server

# Embedding Configuration  
MEMORY_SERVER_EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
MEMORY_SERVER_EMBEDDING_BATCH_SIZE=32
MEMORY_SERVER_EMBEDDING_MAX_SEQ_LENGTH=512

# Server Configuration
MEMORY_SERVER_HOST=localhost
MEMORY_SERVER_PORT=8000
MEMORY_SERVER_LOG_LEVEL=INFO
```

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_SERVER_DATA_DIR` | `.memory-server` | Base directory for data storage |
| `MEMORY_SERVER_EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `MEMORY_SERVER_EMBEDDING_BATCH_SIZE` | `32` | Batch size for embedding generation |
| `MEMORY_SERVER_LOG_LEVEL` | `INFO` | Logging level |

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Claude        │    │  Memory Server   │    │  Storage        │
│   Desktop       │◄──►│                  │◄──►│                 │
│                 │    │  - FastMCP       │    │ - ChromaDB      │
│  Other MCP      │    │  - Tools         │    │ - Embeddings    │
│  Clients        │    │  - Resources     │    │ - Metadata      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Components

- **FastMCP Server**: Handles MCP protocol communication
- **Vector Store**: ChromaDB for semantic search capabilities  
- **Embedding Service**: Local sentence-transformers for text encoding
- **Memory Models**: Pydantic models for data validation
- **Configuration**: Environment-based settings management

## 🔌 Integration Examples

### Claude Desktop
The server is designed to work seamlessly with Claude Desktop via the MCP protocol over stdio transport.

### Browser Agents
HTTP/SSE transport support planned for browser-based AI agents.

### API Integration
Direct Python integration possible:

```python
# Future API example
from memory_server.client import MemoryClient

async with MemoryClient() as client:
    # Add memory
    result = await client.add_memory(
        content="User prefers dark mode for all applications",
        tags=["preference", "ui", "accessibility"]
    )
    
    # Search memories
    results = await client.search("user interface preferences")
```

## 📁 Data Storage

- **Default Location**: `.memory-server/`
- **Vector Database**: `.memory-server/chroma/`
- **Logs**: Console output (configurable)

All data is stored locally on your machine. No data is sent to external services.

## 🛡️ Privacy & Security

- **Local-First**: All processing happens on your machine
- **No Telemetry**: ChromaDB telemetry is disabled
- **No External Calls**: Embeddings generated locally
- **Configurable**: Full control over data storage location

## 🚧 Development

### Project Structure

```
memory-for-ai/
├── src/memory_server/
│   ├── main.py              # FastMCP server entry point
│   ├── models/              # Pydantic data models
│   ├── storage/             # Storage backends (ChromaDB)
│   ├── search/              # Embedding services
│   └── mcp/                 # MCP-specific implementations
├── examples/                # Integration examples
├── tests/                   # Test suite
└── docs/                    # Documentation
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Roadmap

### Current (MVP)
- ✅ Basic memory storage and retrieval
- ✅ Semantic search with embeddings
- ✅ Claude Desktop integration
- ✅ MCP protocol compliance

### Planned Features
- [ ] HTTP/SSE transport for web clients
- [ ] Memory collections and organization
- [ ] Advanced search filters and ranking
- [ ] Memory importance scoring
- [ ] Bulk import/export functionality
- [ ] Web dashboard for memory management
- [ ] Plugin architecture for extensibility

## 🐛 Troubleshooting

### Common Issues

**Server won't start**
- Check Python version (3.12+ required)
- Verify UV is installed and up to date
- Check `.env` file format

**Claude Desktop not connecting**
- Verify `claude_desktop_config.json` path is correct
- Check that `cwd` points to your project directory
- Restart Claude Desktop after config changes

**Import errors**
- Run `uv sync` to ensure dependencies are installed
- Check that you're in the correct directory

### Logs

Server logs include:
- Configuration loading
- Embedding model initialization
- Vector store setup
- MCP tool calls and responses

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Anthropic](https://www.anthropic.com) for the MCP specification
- [ChromaDB](https://www.trychroma.com/) for the vector database
- [sentence-transformers](https://www.sbert.net/) for embeddings
- [UV](https://docs.astral.sh/uv/) for package management

---

**Made with ❤️ (and Cursor) for the AI community**
