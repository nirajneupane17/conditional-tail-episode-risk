# Conditional Tail Episode Risk -- reproduction pipeline
PY = python

.PHONY: all benchmarks cter incremental simulation stress figures clean

all: benchmarks cter incremental simulation stress figures ## run the full pipeline

benchmarks:   ## VaR/ES benchmarks + FZ0 scoring
	$(PY) src/01_benchmarks.py
cter:         ## CTER occurrence + severity walk-forward (writes deltas)
	$(PY) src/02_cter_pipeline.py
incremental: cter  ## incremental information test (Diebold-Mariano + bootstrap)
	$(PY) src/03_incremental_test.py
simulation:   ## Monte Carlo dependence experiment
	$(PY) src/04_simulation.py
stress:       ## adversarial stress tests
	$(PY) src/05_stress_tests.py
figures:      ## all figures
	$(PY) src/06_figures.py

clean:        ## remove generated intermediates
	rm -f results/incremental_deltas.npz
	find . -type d -name __pycache__ -exec rm -rf {} +
