PY ?= python3
SMOKE ?= /tmp/kd_smoke

.PHONY: help install lint format test smoke

help:
	@echo "make install  - install the package and dev tools (editable)"
	@echo "make lint     - check code style with ruff"
	@echo "make format   - auto-fix style with ruff"
	@echo "make test     - unit tests (resume, cache reuse, label budgets), ~2 min"
	@echo "make smoke    - run every script on tiny synthetic data (no downloads, CPU, ~2 min)"

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check .

test:
	$(PY) -m pytest -q tests

format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

smoke:
	$(PY) scripts/feasibility.py --source dummy --device cpu --student tiny \
		--n-train 32 --n-val 16 --steps 6 \
		--out $(SMOKE)/feasibility --cache-dir $(SMOKE)/cache --no-cache
	$(PY) scripts/fewlabel_v2.py --data dummy --device cpu --student tiny \
		--pool 32 --n-val 16 --labeled 4 --steps 6 --eval-every 3 \
		--out $(SMOKE)/fewlabel_v2 --cache-dir $(SMOKE)/cache --no-cache
	$(PY) scripts/fewlabel_v1.py --data dummy --device cpu --student tiny \
		--pool 32 --n-val 16 --labeled 4 --steps 6 --eval-every 3 \
		--out $(SMOKE)/fewlabel_v1 --cache-dir $(SMOKE)/cache --no-cache
