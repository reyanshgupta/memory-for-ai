@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\..\"
set "CONFIG_DIR=%USERPROFILE%\.config\claude-code"
set "CONFIG_FILE=%CONFIG_DIR%\mcp.json"

echo 🧠 Setting up Memory for AI MCP Server for Claude Code (Windows)
echo ==================================================

where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ UV is not installed. Please install UV first:
    echo    https://docs.astral.sh/uv/getting-started/installation/
    echo    Or use: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    exit /b 1
)

echo ✅ UV found

cd /d "%PROJECT_ROOT%"

echo 📦 Installing dependencies...
uv sync

echo 📁 Creating Claude Code config directory...
if not exist "%CONFIG_DIR%" mkdir "%CONFIG_DIR%"

echo ⚙️  Configuring MCP server...

(
echo {
echo   "memory-for-ai": {
echo     "command": "uv",
echo     "args": ["run", "memory-server"^],
echo     "cwd": "%PROJECT_ROOT:\=\\%",
echo     "env": {}
echo   }
echo }
) > "%CONFIG_FILE%"

echo ✅ Setup complete!
echo.
echo 📋 Configuration details:
echo    Config file: %CONFIG_FILE%
echo    Project root: %PROJECT_ROOT%
echo.
echo 🚀 To use the Memory for AI server:
echo    1. Restart Claude Code if it's running
echo    2. The MCP server will be available as 'memory-for-ai'
echo    3. Use tools like add_memory, search_memories, etc.
echo.
echo 🔧 To test the server manually:
echo    cd "%PROJECT_ROOT%" ^&^& uv run memory-server

pause