# Convenience targets. Every one is a single plain command - if `make` is unavailable
# (e.g. on Windows), run the command shown in the recipe directly.

IMAGE ?= care-agent

.PHONY: help install verify test run eval docker-build docker-eval clean

help:
	@echo "install       install dependencies and the package"
	@echo "verify        validate the policy pack (schema + safety invariants)"
	@echo "test          run the test suite"
	@echo "run           run one demo session through the agent"
	@echo "eval          run the full evaluation suite and write report.json/report.md"
	@echo "docker-build  build the container image"
	@echo "docker-eval   run the evaluation in the container, writing reports to the host"

install:
	pip install -r requirements.txt
	pip install -e .

verify:
	python scripts/verify_policy.py

test:
	python -m pytest -q

run:
	python -m care_agent.cli

eval:
	python run_all.py

docker-build:
	docker build -t $(IMAGE) .

docker-eval: docker-build
	docker run --rm -v "$(CURDIR)":/out $(IMAGE) \
		python run_all.py --json /out/report.json --md /out/report.md

clean:
	rm -rf .pytest_cache src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
