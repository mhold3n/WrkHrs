# AI Stack Makefile
# Provides common targets for development and production deployment

.PHONY: help build build-dev build-prod up up-dev up-prod down logs clean test health status pull-models setup-dev setup-prod push-images

# Default target
.DEFAULT_GOAL := help

# Variables
COMPOSE_BASE = -f compose/docker-compose.base.yml
COMPOSE_DEV = $(COMPOSE_BASE) -f compose/docker-compose.dev.yml
COMPOSE_PROD = $(COMPOSE_BASE) -f compose/docker-compose.prod.yml
ENV_FILE = .env

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)AI Stack Management$(NC)"
	@echo "==================="
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Environment files:$(NC)"
	@echo "  Create $(YELLOW).env$(NC) from $(YELLOW).env.example$(NC) before running"

setup: ## Initial setup - create .env file
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "$(YELLOW)Creating .env file from template...$(NC)"; \
		cp .env.example $(ENV_FILE); \
		echo "$(GREEN)✓ .env file created$(NC)"; \
		echo "$(YELLOW)Please review and modify .env as needed$(NC)"; \
	else \
		echo "$(GREEN)✓ .env file already exists$(NC)"; \
	fi

# Build targets
build: setup ## Build all services for current environment
	@echo "$(BLUE)Building all services...$(NC)"
	docker compose $(COMPOSE_DEV) build

build-dev: setup ## Build all services for development
	@echo "$(BLUE)Building development services...$(NC)"
	docker compose $(COMPOSE_DEV) build

build-prod: setup ## Build all services for production
	@echo "$(BLUE)Building production services...$(NC)"
	docker compose $(COMPOSE_PROD) build

build-service: setup ## Build specific service (usage: make build-service SERVICE=gateway-api)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE parameter required$(NC)"; \
		echo "Usage: make build-service SERVICE=gateway-api"; \
		exit 1; \
	fi
	@echo "$(BLUE)Building service: $(SERVICE)$(NC)"
	docker compose $(COMPOSE_DEV) build $(SERVICE)

# Deployment targets
up: up-dev ## Start all services (default: development)

up-dev: setup build-dev ## Start all services in development mode
	@echo "$(BLUE)Starting development environment...$(NC)"
	docker compose $(COMPOSE_DEV) up -d
	@echo "$(GREEN)✓ Development environment started$(NC)"
	@echo "$(YELLOW)Gateway API:$(NC) http://localhost:8080"
	@echo "$(YELLOW)Orchestrator:$(NC) http://localhost:8081"
	@echo "$(YELLOW)RAG API:$(NC) http://localhost:8082"
	@echo "$(YELLOW)Qdrant:$(NC) http://localhost:6333"

up-prod: setup build-prod ## Start all services in production mode
	@echo "$(BLUE)Starting production environment...$(NC)"
	@echo "$(YELLOW)Note: Ensure NVIDIA drivers and container toolkit are installed$(NC)"
	docker compose $(COMPOSE_PROD) up -d
	@echo "$(GREEN)✓ Production environment started$(NC)"
	@echo "$(YELLOW)Gateway API:$(NC) http://localhost:8080"
	@echo "$(YELLOW)vLLM API:$(NC) http://localhost:8001"

down: ## Stop all services
	@echo "$(BLUE)Stopping all services...$(NC)"
	docker compose $(COMPOSE_DEV) down 2>/dev/null || true
	docker compose $(COMPOSE_PROD) down 2>/dev/null || true
	@echo "$(GREEN)✓ All services stopped$(NC)"

restart: down up ## Restart all services

restart-service: ## Restart specific service (usage: make restart-service SERVICE=gateway-api)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE parameter required$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Restarting service: $(SERVICE)$(NC)"
	docker compose $(COMPOSE_DEV) restart $(SERVICE)

# Logging and monitoring
logs: ## Show logs for all services
	docker compose $(COMPOSE_DEV) logs -f

logs-service: ## Show logs for specific service (usage: make logs-service SERVICE=gateway-api)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE parameter required$(NC)"; \
		exit 1; \
	fi
	docker compose $(COMPOSE_DEV) logs -f $(SERVICE)

health: ## Check health of all services
	@echo "$(BLUE)Checking service health...$(NC)"
	@echo ""
	@services="gateway-api orchestrator tool-registry mcp rag-api asr-api"; \
	for service in $$services; do \
		port=$$(docker compose $(COMPOSE_DEV) port $$service 8000 2>/dev/null | cut -d: -f2); \
		if [ -n "$$port" ]; then \
			if curl -s -f http://localhost:$$port/health > /dev/null 2>&1; then \
				echo "$(GREEN)✓ $$service$(NC) - healthy"; \
			else \
				echo "$(RED)✗ $$service$(NC) - unhealthy"; \
			fi; \
		else \
			echo "$(YELLOW)? $$service$(NC) - not running"; \
		fi; \
	done
	@echo ""
	@if curl -s -f http://localhost:6333/ready > /dev/null 2>&1; then \
		echo "$(GREEN)✓ qdrant$(NC) - ready"; \
	else \
		echo "$(RED)✗ qdrant$(NC) - not ready"; \
	fi
	@if curl -s -f http://localhost:11434/api/tags > /dev/null 2>&1; then \
		echo "$(GREEN)✓ ollama$(NC) - ready"; \
	else \
		echo "$(YELLOW)? ollama$(NC) - not ready (check if running)"; \
	fi

status: ## Show status of all containers
	@echo "$(BLUE)Container status:$(NC)"
	docker compose $(COMPOSE_DEV) ps

# Model management
pull-models: ## Pull/download required models
	@echo "$(BLUE)Pulling Ollama model...$(NC)"
	@model=$$(grep OLLAMA_MODEL $(ENV_FILE) | cut -d'=' -f2); \
	if [ -n "$$model" ]; then \
		echo "Pulling model: $$model"; \
		curl -X POST http://localhost:11434/api/pull -d '{"name":"'$$model'"}'; \
	else \
		echo "$(YELLOW)No OLLAMA_MODEL found in .env$(NC)"; \
	fi

# Development helpers
shell: ## Open shell in specific service (usage: make shell SERVICE=gateway-api)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE parameter required$(NC)"; \
		exit 1; \
	fi
	docker compose $(COMPOSE_DEV) exec $(SERVICE) /bin/bash

shell-root: ## Open root shell in specific service (usage: make shell-root SERVICE=gateway-api)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE parameter required$(NC)"; \
		exit 1; \
	fi
	docker compose $(COMPOSE_DEV) exec -u root $(SERVICE) /bin/bash

# Testing
test: ## Run basic API tests
	@echo "$(BLUE)Running basic API tests...$(NC)"
	@echo "Testing Gateway API..."
	@if curl -s -f http://localhost:8080/health | jq -r '.status' | grep -q "healthy"; then \
		echo "$(GREEN)✓ Gateway API$(NC)"; \
	else \
		echo "$(RED)✗ Gateway API$(NC)"; \
	fi
	@echo "Testing Orchestrator..."
	@if curl -s -f http://localhost:8081/health | jq -r '.status' | grep -q "healthy"; then \
		echo "$(GREEN)✓ Orchestrator$(NC)"; \
	else \
		echo "$(RED)✗ Orchestrator$(NC)"; \
	fi

test-chat: ## Test chat endpoint
	@echo "$(BLUE)Testing chat endpoint...$(NC)"
	curl -X POST http://localhost:8080/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"messages":[{"role":"user","content":"What is the strength of steel?"}],"model":"test","temperature":0.7}' \
		| jq .

# Data management
init-data: ## Initialize with sample data
	@echo "$(BLUE)Initializing sample data...$(NC)"
	@echo "Adding sample document to RAG..."
	curl -X POST http://localhost:8082/documents \
		-H "Content-Type: application/json" \
		-d '{"content":"Steel is a metallic alloy composed primarily of iron and carbon. It has high tensile strength and is widely used in construction and manufacturing.","domain":"materials","source":"sample"}' \
		| jq .
	@echo "$(GREEN)✓ Sample data added$(NC)"

backup-data: ## Backup persistent data
	@echo "$(BLUE)Creating data backup...$(NC)"
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	mkdir -p backups; \
	docker compose $(COMPOSE_DEV) exec -T qdrant tar -czf - /qdrant/storage | cat > backups/qdrant_$$timestamp.tar.gz; \
	echo "$(GREEN)✓ Backup created: backups/qdrant_$$timestamp.tar.gz$(NC)"

# Cleanup
clean: down ## Clean up containers, networks, and volumes
	@echo "$(BLUE)Cleaning up containers and networks...$(NC)"
	docker compose $(COMPOSE_DEV) down -v --remove-orphans
	docker compose $(COMPOSE_PROD) down -v --remove-orphans
	@echo "$(GREEN)✓ Cleanup completed$(NC)"

clean-images: ## Remove built images
	@echo "$(BLUE)Removing built images...$(NC)"
	docker images --filter "reference=ai-stack*" -q | xargs -r docker rmi
	@echo "$(GREEN)✓ Images removed$(NC)"

clean-all: clean clean-images ## Full cleanup including images
	@echo "$(GREEN)✓ Full cleanup completed$(NC)"

# Production deployment helpers
push-images: ## Push images to registry (requires REGISTRY env var)
	@if [ -z "$(REGISTRY)" ]; then \
		echo "$(RED)Error: REGISTRY environment variable required$(NC)"; \
		echo "Usage: make push-images REGISTRY=ghcr.io/username"; \
		exit 1; \
	fi
	@echo "$(BLUE)Pushing images to $(REGISTRY)...$(NC)"
	@services="gateway orchestrator tool-registry mcp rag asr"; \
	for service in $$services; do \
		echo "Pushing $$service..."; \
		docker tag ai-stack-$$service $(REGISTRY)/$$service:latest; \
		docker push $(REGISTRY)/$$service:latest; \
	done
	@echo "$(GREEN)✓ All images pushed$(NC)"

deploy-prod: ## Deploy to production (after pushing images)
	@echo "$(BLUE)Deploying to production...$(NC)"
	@echo "$(YELLOW)Ensure .env is configured for production$(NC)"
	@echo "$(YELLOW)Ensure GPU drivers are installed$(NC)"
	$(MAKE) setup
	$(MAKE) build-prod
	$(MAKE) up-prod
	@echo "$(GREEN)✓ Production deployment completed$(NC)"

# Development workflow
dev-setup: setup build-dev up-dev pull-models init-data ## Complete development setup
	@echo "$(GREEN)✓ Development environment ready!$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  - Check service health: make health"
	@echo "  - View logs: make logs"
	@echo "  - Test chat: make test-chat"

# Monitoring
monitor: ## Show real-time resource usage
	@echo "$(BLUE)Monitoring containers (Ctrl+C to exit)...$(NC)"
	docker stats $$(docker compose $(COMPOSE_DEV) ps -q)

# Plugin management
refresh-plugins: ## Refresh tool registry plugins
	@echo "$(BLUE)Refreshing plugins...$(NC)"
	curl -X POST http://localhost:8086/plugins/refresh | jq .

list-plugins: ## List available plugins
	@echo "$(BLUE)Available plugins:$(NC)"
	curl -s http://localhost:8086/tools | jq '.tools[] | {name: .name, type: .type, description: .description}'