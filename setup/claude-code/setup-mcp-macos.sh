#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$HOME/.claude.json"

echo "🧠 Setting up Memory for AI MCP Server for Claude Code (macOS)"
echo "=================================================="

if ! command -v uv &> /dev/null; then
    echo "❌ UV is not installed. Please install UV first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ UV found"

cd "$PROJECT_ROOT"

echo "📦 Installing dependencies..."
uv sync

echo "📁 Checking Claude Code config file..."

echo "⚙️  Configuring MCP server..."

# Check if config file exists and has mcpServers section
if [ -f "$CONFIG_FILE" ]; then
    echo "📝 Existing config found, updating..."
    # Create a backup
    cp "$CONFIG_FILE" "$CONFIG_FILE.backup"

    # Check if mcpServers section exists
    if grep -q '"mcpServers"' "$CONFIG_FILE"; then
        # Add memory-for-ai to existing mcpServers section
        python3 -c "
import json
import sys

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = json.load(f)

    if 'mcpServers' not in config:
        config['mcpServers'] = {}

    config['mcpServers']['memory-for-ai'] = {
        'type': 'stdio',
        'command': 'uv',
        'args': ['run', 'memory-server'],
        'cwd': '$PROJECT_ROOT',
        'env': {}
    }

    with open('$CONFIG_FILE', 'w') as f:
        json.dump(config, f, indent=2)

    print('✅ Updated existing config')
except Exception as e:
    print(f'❌ Error updating config: {e}')
    sys.exit(1)
"
    else
        # Add mcpServers section to existing config
        python3 -c "
import json
import sys

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = json.load(f)

    config['mcpServers'] = {
        'memory-for-ai': {
            'type': 'stdio',
            'command': 'uv',
            'args': ['run', 'memory-server'],
            'cwd': '$PROJECT_ROOT',
            'env': {}
        }
    }

    with open('$CONFIG_FILE', 'w') as f:
        json.dump(config, f, indent=2)

    print('✅ Added mcpServers section to existing config')
except Exception as e:
    print(f'❌ Error updating config: {e}')
    sys.exit(1)
"
    fi
else
    echo "📝 Creating new config file..."
    cat > "$CONFIG_FILE" << EOF
{
  "mcpServers": {
    "memory-for-ai": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "memory-server"],
      "cwd": "$PROJECT_ROOT",
      "env": {}
    }
  }
}
EOF
fi

echo "✅ Setup complete!"
echo ""
echo "📋 Configuration details:"
echo "   Config file: $CONFIG_FILE"
echo "   Project root: $PROJECT_ROOT"
echo ""
echo "🚀 To use the Memory for AI server:"
echo "   1. Restart Claude Code if it's running"
echo "   2. The MCP server will be available as 'memory-for-ai'"
echo "   3. Use tools like add_memory, search_memories, etc."
echo ""
echo "🔧 To test the server manually:"
echo "   cd '$PROJECT_ROOT' && uv run memory-server"