# Convenience targets. Everything here is a thin wrapper around main.py or a
# standard tool - nothing is hidden behind make.

PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin

.DEFAULT_GOAL := help
.PHONY: help venv install install-dev check test lint format typecheck \
        config fetch scan run backtest optimize selftest telegram-test \
        docker-build docker-up docker-down docker-logs clean clean-cache

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the virtualenv
	$(PYTHON) -m venv $(VENV)

install: venv  ## Install runtime dependencies
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

install-dev: venv  ## Install runtime + development dependencies
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt

check: lint test  ## Lint and test - run this before every commit

test:  ## Run the test suite
	$(BIN)/pytest -q

lint:  ## Static checks
	$(BIN)/ruff check .

format:  ## Auto-fix what can be auto-fixed
	$(BIN)/ruff check . --fix
	$(BIN)/ruff format .

typecheck:  ## Type checking (advisory)
	$(BIN)/mypy config data indicators analysis strategies risk backtest notify live utils

# --- bot commands ----------------------------------------------------------
config:  ## Validate the configuration
	$(BIN)/python main.py config-check

fetch:  ## Download and cache candles for every configured symbol
	$(BIN)/python main.py fetch

scan:  ## One analysis pass, printed
	$(BIN)/python main.py scan

run:  ## Start the live signal loop
	$(BIN)/python main.py run

backtest:  ## Replay history (SYMBOL=BTC/USDT TF=15m)
	$(BIN)/python main.py backtest $(if $(SYMBOL),--symbol $(SYMBOL)) $(if $(TF),--timeframe $(TF))

optimize:  ## Walk-forward parameter search
	$(BIN)/python main.py optimize

selftest:  ## Exercise the pipeline offline on synthetic data
	$(BIN)/python main.py selftest

telegram-test:  ## Verify Telegram notifications
	$(BIN)/python main.py telegram-test

# --- docker ----------------------------------------------------------------
docker-build:  ## Build the image
	docker compose build

docker-up:  ## Start the bot in the background
	docker compose up -d

docker-down:  ## Stop the bot
	docker compose down

docker-logs:  ## Follow the logs
	docker compose logs -f bot

# --- housekeeping ----------------------------------------------------------
clean:  ## Remove caches and build artefacts (keeps logs and candle data)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info

clean-cache:  ## Also delete cached candles and reports (NOT logs/state)
	rm -rf data/cache reports
