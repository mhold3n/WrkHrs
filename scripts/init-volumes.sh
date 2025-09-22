#!/bin/bash
# Initialize data directories for AI Stack volumes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Initializing AI Stack data directories...${NC}"

# Source environment variables if .env exists
if [ -f .env ]; then
    source .env
    echo -e "${GREEN}✓ Loaded environment variables from .env${NC}"
else
    echo -e "${YELLOW}⚠ No .env file found, using default paths${NC}"
fi

# Set default paths if not defined in environment
QDRANT_DATA_PATH=${QDRANT_DATA_PATH:-./data/qdrant}
RAG_CACHE_PATH=${RAG_CACHE_PATH:-./data/rag_cache}
MODELS_PATH=${MODELS_PATH:-./models}
PLUGINS_PATH=${PLUGINS_PATH:-./plugins}
MCP_DATA_PATH=${MCP_DATA_PATH:-./data/mcp}
LOGS_PATH=${LOGS_PATH:-./logs}
GATEWAY_STATE_PATH=${GATEWAY_STATE_PATH:-./data/gateway_state}

# Function to create directory with proper permissions
create_dir() {
    local dir_path="$1"
    local description="$2"
    
    if [ ! -d "$dir_path" ]; then
        mkdir -p "$dir_path"
        echo -e "${GREEN}✓ Created${NC} $description: $dir_path"
    else
        echo -e "${YELLOW}↻ Exists${NC} $description: $dir_path"
    fi
    
    # Ensure proper permissions (readable/writable by docker user)
    chmod 755 "$dir_path"
}

# Create all required directories
echo ""
echo -e "${BLUE}Creating data directories:${NC}"

create_dir "$QDRANT_DATA_PATH" "Qdrant vector database data"
create_dir "$RAG_CACHE_PATH" "RAG service cache"
create_dir "$MODELS_PATH" "LLM models storage"
create_dir "$PLUGINS_PATH" "Tool registry plugins"
create_dir "$MCP_DATA_PATH" "MCP domain data"
create_dir "$LOGS_PATH" "Service logs"
create_dir "$GATEWAY_STATE_PATH" "Gateway state storage"

# Create subdirectories for MCP domains
echo ""
echo -e "${BLUE}Creating MCP domain subdirectories:${NC}"
create_dir "$MCP_DATA_PATH/chemistry" "Chemistry domain data"
create_dir "$MCP_DATA_PATH/mechanical" "Mechanical domain data"
create_dir "$MCP_DATA_PATH/materials" "Materials domain data"

# Create plugin subdirectories
echo ""
echo -e "${BLUE}Creating plugin subdirectories:${NC}"
create_dir "$PLUGINS_PATH/calculators" "Calculator plugins"
create_dir "$PLUGINS_PATH/converters" "Unit converter plugins"
create_dir "$PLUGINS_PATH/cli_tools" "CLI tool wrappers"

# Create logs subdirectories for each service
echo ""
echo -e "${BLUE}Creating service log directories:${NC}"
create_dir "$LOGS_PATH/gateway" "Gateway service logs"
create_dir "$LOGS_PATH/orchestrator" "Orchestrator service logs"
create_dir "$LOGS_PATH/rag" "RAG service logs"
create_dir "$LOGS_PATH/asr" "ASR service logs"
create_dir "$LOGS_PATH/mcp" "MCP service logs"
create_dir "$LOGS_PATH/tool_registry" "Tool registry logs"

# Create backup directories
echo ""
echo -e "${BLUE}Creating backup directories:${NC}"
create_dir "./backups" "Data backups"
create_dir "./backups/qdrant" "Qdrant backups"
create_dir "./backups/configs" "Configuration backups"

# Create a simple .gitkeep file in important directories
echo ""
echo -e "${BLUE}Adding .gitkeep files:${NC}"
for dir in "$QDRANT_DATA_PATH" "$RAG_CACHE_PATH" "$GATEWAY_STATE_PATH" "./backups"; do
    if [ ! -f "$dir/.gitkeep" ]; then
        touch "$dir/.gitkeep"
        echo -e "${GREEN}✓ Added .gitkeep to${NC} $dir"
    fi
done

# Create a basic .gitignore for data directories
echo ""
echo -e "${BLUE}Creating data .gitignore:${NC}"
cat > data/.gitignore << 'EOF'
# AI Stack Data Directories
# Keep structure but ignore contents

# Qdrant vector database files
qdrant/*
!qdrant/.gitkeep

# RAG cache files
rag_cache/*
!rag_cache/.gitkeep

# Gateway state files
gateway_state/*
!gateway_state/.gitkeep

# MCP domain data (except samples)
mcp/*/[!sample]*
!mcp/*/.gitkeep

# Log files (keep recent for debugging)
logs/*
!logs/.gitkeep

EOF

echo -e "${GREEN}✓ Created data/.gitignore${NC}"

# Check disk space
echo ""
echo -e "${BLUE}Checking available disk space:${NC}"
df -h . | awk 'NR==2 {printf "Available: %s (%s used of %s)\n", $4, $5, $2}'

# Create a summary report
echo ""
echo -e "${GREEN}✅ Volume initialization complete!${NC}"
echo ""
echo -e "${YELLOW}Directory summary:${NC}"
echo "  • Qdrant data: $QDRANT_DATA_PATH"
echo "  • RAG cache: $RAG_CACHE_PATH"
echo "  • Models: $MODELS_PATH"
echo "  • Plugins: $PLUGINS_PATH"
echo "  • MCP data: $MCP_DATA_PATH"
echo "  • Logs: $LOGS_PATH"
echo "  • Gateway state: $GATEWAY_STATE_PATH"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Add sample data: make init-data"
echo "  2. Start services: make up-dev"
echo "  3. Check health: make health"
