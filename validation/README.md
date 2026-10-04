# Verification and validation

* **Verification** shows that the solvers solve their equations correctly.
  `run_verification.py` compares them with closed-form results and with
  independent solutions of the same theories (Lanchester-Prandtl, Glauert, Kinner,
  Helmbold), and measures the limits of each method (sweep, taper, mesh,
  ground effect). It computes every number in the report; nothing is
  hard-coded.

  ```bash
  python validation/run_verification.py            # writes docs/verification_report.md
  python validation/run_verification.py out.md     # writes another file
  ```

  `tests/test_analytical_verification.py` checks the same items with fixed
  tolerances.

* **Validation** compares the models with experiments. It needs measured data
  with full provenance. The `experimental/` folder holds those datasets; it is
  empty at this time. See `experimental/README.md` for the format and the
  candidate sources.

The references are in `ventorum/reference/analytic.py` (with citations) and
the loader with the provenance check is in
`ventorum/reference/experimental.py`.
