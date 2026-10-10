# Verification and validation

- **Verification** checks that the solvers solve their equations correctly: comparison with closed-form and independent solutions, grid convergence and the image method. `python validation/run_verification.py` writes the report below.
- **Validation** compares the results with experiments and higher-order methods. It needs the reference database, with full provenance and quality grades, and is not done yet.
- **Comparison with AVL**: the harness in `validation/avl/` runs the same cases in Ventorum and in AVL (used as a black box: input files in, output files out) and writes tables of the force, moment and static-derivative differences plus timing studies. The comparison results are private and are not in the documentation yet.

```{toctree}
:maxdepth: 1

../verification_report
```
