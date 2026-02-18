ifeq (,$(wildcard .env))
$(error .env file is missing. Please create one based on .env.example)
endif

include .env

CHECK_DIRS := .

# --- QA ---

format-fix:
	uv run ruff format $(CHECK_DIRS)
	uv run ruff check --select I --fix

lint-fix:
	uv run ruff check --fix

format-check:
	uv run ruff format --check $(CHECK_DIRS)
	uv run ruff check -e
	uv run ruff check --select I -e

lint-check:
	uv run ruff check $(CHECK_DIRS)

# --- Agent Config (from .env) ---
# Set AGENT_ID and AGENT_ARN in your .env file from CDK stack outputs

EVAL_OUTPUT_DIR ?= evaluation_results

# --- Evaluation ---

## Run the full evaluation suite (all test cases + built-in evaluators)
eval:
	@test -n "$(AGENT_ID)" || (echo "Error: AGENT_ID is required. Set it in .env" && exit 1)
	@test -n "$(AGENT_ARN)" || (echo "Error: AGENT_ARN is required. Set it in .env" && exit 1)
	uv run python -m src.evaluation.runner \
		--agent-id $(AGENT_ID) \
		--agent-arn $(AGENT_ARN) \
		--output-dir $(EVAL_OUTPUT_DIR)

## Run evaluation for specific categories (usage: make eval-categories CATEGORIES="basic_search dietary_search")
CATEGORIES ?= basic_search
eval-categories:
	@test -n "$(AGENT_ID)" || (echo "Error: AGENT_ID is required. Set it in .env" && exit 1)
	@test -n "$(AGENT_ARN)" || (echo "Error: AGENT_ARN is required. Set it in .env" && exit 1)
	uv run python -m src.evaluation.runner \
		--agent-id $(AGENT_ID) \
		--agent-arn $(AGENT_ARN) \
		--categories $(CATEGORIES) \
		--output-dir $(EVAL_OUTPUT_DIR)

## Evaluate an existing session (usage: make eval-session SESSION_ID=your-session-id)
eval-session:
	@test -n "$(SESSION_ID)" || (echo "Error: SESSION_ID is required. Usage: make eval-session SESSION_ID=your-session-id" && exit 1)
	@test -n "$(AGENT_ID)" || (echo "Error: AGENT_ID is required. Set it in .env" && exit 1)
	uv run python -m src.evaluation.on_demand \
		--session-id $(SESSION_ID) \
		--agent-id $(AGENT_ID) \
		--output-dir $(EVAL_OUTPUT_DIR)

## Setup online evaluation for production monitoring (usage: make eval-online SAMPLING_RATE=10)
SAMPLING_RATE ?= 10
eval-online:
	@test -n "$(AGENT_ID)" || (echo "Error: AGENT_ID is required. Set it in .env" && exit 1)
	uv run python -m src.evaluation.online \
		--agent-id $(AGENT_ID) \
		--sampling-rate $(SAMPLING_RATE) \
		--no-custom

## Run only safety-related evaluations (safety + out_of_scope categories)
eval-safety:
	@test -n "$(AGENT_ID)" || (echo "Error: AGENT_ID is required. Set it in .env" && exit 1)
	@test -n "$(AGENT_ARN)" || (echo "Error: AGENT_ARN is required. Set it in .env" && exit 1)
	uv run python -m src.evaluation.runner \
		--agent-id $(AGENT_ID) \
		--agent-arn $(AGENT_ARN) \
		--categories safety out_of_scope \
		--output-dir $(EVAL_OUTPUT_DIR)
