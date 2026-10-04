# Performance architecture of the solve pipeline

This page explains how a single solve and a sweep (batch) of the lattice solvers (vortex lattice, linear and nonlinear lifting line) and of the Fourier solver avoid repeated work. It is for developers: each mechanism has rules that a change must keep. The measured times are not here (study results stay private); the tests check that the mechanisms give the same results.

## The pipeline of one solve

`LatticeSolver.solve` (`ventorum/solvers/lattice_base.py`):

1. Input checks (`validate_aircraft`).
2. **Surfaces fingerprint** (`lattice_cache.surfaces_fingerprint`), computed once per solve.
3. **Geometry information** (`lattice_cache.geometry_info`): main-surface index and automatic reference values, cached by the fingerprint. `Aircraft.compute_reference_values(auto=...)` applies them with the usual rule (a value that the user set is kept).
4. **Lattice** (`build`): taken from the lattice cache, or built.
5. **Circulation** (`solve_circulation`): the system and the dense solve.
6. **Loads** (`compute_loads`): near-field forces and moments, profile drag, section moments, Trefftz-plane induced drag.
7. Trust score and the result object.

The Fourier solver (`ventorum/solvers/fourier.py`) uses steps 1 to 3 and caches its whole system per geometry (see below).

## Caches

| Cache | Where | Key | Holds | Rules |
| --- | --- | --- | --- | --- |
| Lattice cache | `ventorum/geometry/lattice_cache.py` | builder helper functions (the objects), surfaces fingerprint, collocation, chordwise count and spacing, spanwise mesh settings | the last 8 built lattices | arrays are read-only; `get_or_build` returns a shallow copy with `kernel_cache = None` |
| Geometry information | same module | surfaces fingerprint | main-surface index, automatic reference values | depends only on the surfaces |
| `lattice.geom_cache` | field of `VortexLattice` | per item | airfoil groups, symmetry geometry of the force points, Trefftz-plane core data, left panels of the symmetric map, unknown maps, lifting-line strip data and airfoil groups per map, quarter-chord sweep, main-surface strip count, largest chord | lattice-only data; shared by the shallow copies of one cached lattice; the cached unknown maps and strip data are read-only |
| `lattice.kernel_cache` | field of `VortexLattice` | per item | the part of the influence matrices and velocity tensors from the bound vortices and the chordwise legs (they do not change with the wake direction) | set only by vortex-lattice sweeps (`solve_sweep`, the parallel `alpha_sweep`); None in a single solve and in a lifting-line batch, so that these always give the same bits |
| Fourier basis | `fourier._fourier_basis` | number of terms N | collocation and quadrature angles, sine tables | read-only |
| Fourier system | `fourier._fourier_system` | surfaces fingerprint, N, helper functions | section data, system matrix, quadrature data, exact span integrals | read-only; the moment reference point is applied per call |
| Gauss nodes | `fourier._gauss_legendre` | order | nodes and weights | read-only |

### The fingerprint

`_fingerprint` walks every public field (names that do not start with `_`) of the surfaces, the sections and the airfoils, generically, so a new field is covered without a change here. Arrays enter with their dtype, shape and bytes. An object of an unknown type enters with its identity (a change inside it is not seen).

Rules for a change:

* A new input of `build_lattice` that is not a field of the surfaces or one of the mesh settings in `lattice_key` must be added to the key.
* A new module-level helper that `build_lattice` calls must be added to `_BUILD_HELPERS`. The key holds the function objects themselves, not their `id`: code that replaces a helper (the verification case V10 replaces `core_groups` to force the core groups) must never get a lattice built with the original. Python can give the `id` of a freed function to a new one, so an `id` is not enough (this was a bug, see `tests/test_lattice_cache.py`).
* Never write into the arrays of a lattice. The arrays of a cached lattice are read-only; a write raises `ValueError`.
* Put data into `geom_cache` only if it depends on the lattice alone (not on the flight condition, the wake direction or the ground).

## Dense solves: BLAS on one thread

`utils.parallel.blas_single_thread` limits BLAS and LAPACK to one thread (with `threadpoolctl`, a dependency). `solve_threads` applies it in every lattice solve for every kernel backend, and the Fourier solver applies it around its solves. Reason: for the matrix sizes of Ventorum (up to a few hundred unknowns) a multi-threaded OpenBLAS solve was many times slower than a solve on one thread on the owner's machine. The tuner may choose a larger count for large systems later.

## Velocity per unknown inside the kernels

`vortex.velocity_tensor_unknowns` gives the velocity at the control points per unit circulation of each unknown, shape `(m, n_unknowns, 3)`. With the Numba and Cython backends, `velocity_unknowns_kernel` adds each source (mirror half, ground image) into its unknown while it computes the velocities, in source order, so the tensor per source is never stored. It equals the tensor-plus-fold path of the same backend to the last bit (`test_velocity_unknowns_kernel_parity`). The other backends compute the tensor per source and fold it with `_fold_columns`, which adds the sources of each unknown in the same order as `np.add.at`, with vectorised gathers.

## One kernel pass in the lifting line

On a lifting-line lattice (`collocation="llt"`) the force points of the loads are the control points of the system. The linear and nonlinear lifting lines therefore build their system from the velocity tensor per unknown and return the induced velocity at the control points in `SolveInfo.v_control`; `compute_loads(v_control=...)` uses it and skips its own kernel evaluation. Conditions: no leg forces, and not in a sweep (sweeps keep the influence matrix with the kernel cache). The nonlinear lifting line also takes its linear start solution from the same tensor. The vortex lattice keeps its own path (its force points are not its control points).

## Lifting-line batches (sweeps)

`LatticeSolver.solve_sweep` of the linear and nonlinear lifting lines calls `solve_batch`, which solves all angles on one lattice (the lifting-line lattice has one chordwise panel, so it does not depend on the attitude, also in ground effect):

1. Setup per angle (`_case_setup`): ground plane, validity checks, notes, wake direction.
2. **One kernel call for all angles** (`core.llt_velocity_tensors`): the source horseshoes of all angles go into one set, and the unknowns of angle k get the columns `k * n + j`. Each column adds its own sources in the same order as a call for one angle, so the tensor of each angle has the same bits as a single solve. The tensors are views into the kernel output (no copy). Large batches are split so that one call stays below `_BATCH_TENSOR_BYTES`.
3. Linear lifting line (`core.solve_llt_linear_batch`): the systems are built per angle and solved with **one call of the dense solver** for all angles (a stacked `numpy.linalg.solve` gives the same bits as one call per matrix). Nonlinear lifting line: Newton's method per angle, in order (continuation out of ground effect), with the tensor of the batch; the restarts reuse that tensor.
4. **Loads of all angles together** (`loads.compute_loads_batch`, `loads.trefftz_induced_drag_batch`): every array operation acts on all angles. `compute_loads` and `trefftz_induced_drag` are the batch functions with one case, so a single solve and a batch share one code path.
5. Result per angle (`_case_result`): trust score, details, warnings. `execution_time` is the share of the angle in the batch time.

Rules for a change:

* Each angle of a batch must give the same bits as a single solve (`tests/test_sweep_batch.py`). Use only operations whose result for one case does not depend on the other cases: element-wise operations, reductions along an axis of one case, `einsum` per case, stacked `numpy.linalg.solve`. Matrix products that BLAS computes (`@` with a matrix) stay per case: a batched product can round differently.
* Trigonometric functions of the angle stay scalar per case (a vectorised `sin` can round differently from the scalar one).
* The per-case contractions (`einsum`) of large systems run on the kernel threads (`core._map_cases`; numpy releases the GIL in `einsum`). The dense solves do not: OpenBLAS calls from several threads run one after the other.
* A batch that runs alone uses all cores for its kernel calls (`solve_threads(n_panels, batch=K)`); in a worker it keeps the threads of the worker. The thread count does not change the results.
* `alpha_sweep` (and `Ventorum.analyze_sweep`) sends lifting-line sweeps to `solve_sweep`; the agent polar tool solves lifting-line polars out of ground effect with `solve_batch(..., continuation=False)`, so each angle equals `vt.analyze`.
* A new per-case value in the loads must be computed for all cases with the rules above, or in the per-case loop at the end of `compute_loads_batch`.

## Small numerical rules

* `utils.vec.cross3` replaces `numpy.cross` on small arrays: the same formula and order of operations (the same bits), without the axis handling of `numpy.cross`.
* The Newton Jacobian of the nonlinear lifting line uses the scalar triple product `u_i . (v_ij x dl_i) = v_ij . (dl_i x u_i)`: one cross product per strip instead of one per pair.
* The symmetry test of the loads compares `|a - b| <= atol` directly (the same answer as `np.allclose` with `rtol = 0` for finite data).

## How to check a change

1. The verification report must not change (`validation/run_verification.py`).
2. The parity tests in `tests/test_kernels.py`, `tests/test_solve_overhead.py`, `tests/test_lattice_cache.py` and `tests/test_sweep_batch.py`.
3. A repeated solve must give the same bits as the first solve (`test_repeated_solve_gives_the_same_bits`).
4. Timing: run on a quiet machine, with no agent or other heavy program, and compare the old and the new code side by side, alternating. A timing of the old code taken at another time is not a reference.
