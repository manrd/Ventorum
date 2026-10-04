# Verification and validation

- **Verification** checks that the solvers solve their equations correctly: comparison with closed-form and independent solutions, grid convergence and the image method. `python validation/run_verification.py` writes the report below.
- **Validation** compares the results with experiments and higher-order methods. It needs the reference database, with full provenance and quality grades, and is not done yet.

```{toctree}
:maxdepth: 1

../verification_report
```
