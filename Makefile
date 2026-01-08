.PHONY: help setup install test validate backup start stop status health clean lint format logs check-deps

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Run initial setup (venv, deps, .env)
	@./setup.sh

install: ## Install/update dependencies
	@pip install --upgrade pip
	@pip install -r requirements.txt

install-dev: ## Install development dependencies
	@pip install --upgrade pip
	@pip install -r requirements.txt
	@if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi

test: ## Run tests
	@pytest tests/ -v

validate: ## Validate configuration
	@python validate_config.py

backup: ## Backup state files
	@./backup_state.sh

reset: ## Reset all bot history, logs, and open trades
	@./reset_all_bots.sh

start: ## Start all bots
	@./start_all_bots.sh

stop: ## Stop all bots
	@./stop_all_bots.sh

status: ## Show bot status
	@python run_bots.py --status

health: ## Start health check server
	@python health_check.py

clean: ## Clean temporary files
	@echo "Cleaning temporary files..."
	@find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo "✓ Clean complete"

lint: ## Run code quality checks
	@echo "Running code quality checks..."
	@if command -v flake8 >/dev/null 2>&1; then \
		echo "Running flake8..."; \
		flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --max-line-length=120 || true; \
	fi
	@if command -v black >/dev/null 2>&1; then \
		echo "Running black check..."; \
		black --check . --line-length 120 || true; \
	fi

format: ## Format code with black
	@if command -v black >/dev/null 2>&1; then \
		black . --line-length 120; \
		echo "✓ Code formatted"; \
	else \
		echo "⚠ black not installed. Run: pip install black"; \
	fi

logs: ## Show recent logs (last 50 lines)
	@echo "Recent log entries:"
	@tail -50 logs/*.log 2>/dev/null || echo "No logs found"

check-deps: ## Check for outdated dependencies
	@pip list --outdated

monitor: ## Monitor bot health
	@if [ -f monitor_bots.sh ]; then ./monitor_bots.sh; else echo "monitor_bots.sh not found"; fi
