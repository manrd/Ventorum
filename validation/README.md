# Verification and validation

* **Verification** shows that the solvers solve their equations correctly.
  `run_verification.py` compares them with closed-form results and with
  independent solutions of the same theories (Lanchester–Prandtl, Glauert, Kinner,
  Helmbold), and measures the limits of each method (sweep, taper, mesh,
  ground effect). It computes every number in the report; nothing is
  hard-coded.

  ```bash
  python validation/run_verification.py            # writes docs/verification_report.md
  python validation/run_verification.py out.md     # writes another file
  ```

  The script writes the report in all cases. It exits with code 1 when a
  table row of the report shows "FAIL". `compare_reports.py` compares a new
  report with the committed one: text must match, numbers may differ by
  round-off, and a residual below 1e-12 may grow to max(1e-15, 100 times
  the committed value) only.

  `tests/test_analytical_verification.py` checks the same items with fixed
  tolerances.

* **Validation** compares the models with experiments. It needs measured data
  with full provenance. The `experimental/` folder holds those datasets; it is
  empty at this time. See `experimental/README.md` for the format and the
  candidate sources. `run_validation.py` runs the Markdown reference cases
  (`reference_template.md`): coefficients against alpha or CL, static
  derivatives such as Cl_beta, and slopes against CL.

* **Speed history**: `python validation/perf_history.py --history PATH`
  appends one timing record per run to a history file outside the
  repository, `--report PATH` prints the trend table, and
  `--compare TREE_A TREE_B` times two source trees side by side,
  alternating (see `docs/design/performance_architecture.md`).

* **GPU gate**: `python validation/gpu_gate.py <log.md>` runs all GPU
  tests and all examples on an NVIDIA GPU and writes a short Markdown
  log for the phase-gate record (see `docs/contributing/index.md`). The
  pytest step has a timeout of 3600 s (`--pytest-timeout SECONDS`). A
  skipped GPU test is not a pass: the log lists the skip reasons, ends with
  "GPU gate: INCOMPLETE" and the exit code is 3.

The references are in `ventorum/reference/analytic.py` (with citations) and
the loader with the provenance check is in
`ventorum/reference/experimental.py`.
