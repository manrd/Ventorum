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
| `lattice.kernel_cache` | field of `VortexLattice` | per item | the part of the influence matrices and velocity tensors from the bound vortices and the chordwise legs (they do not change with the wake direction) | set only by vortex-lattice sweeps out of ground effect (`solve_sweep`, the parallel `alpha_sweep`); None in a single solve, in the agent polar batch and in a lifting-line batch, so that these always give the same bits |
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

On a lifting-line lattice (`collocation="llt"`) the force points of the loads are the control points of the system. The linear and nonlinear lifting lines therefore build their system from the velocity tensor per unknown and return the induced velocity at the control points in `SolveInfo.v_control`; `compute_loads(v_control=...)` uses it and skips its own kernel evaluation. This applies to a single solve and to the lifting-line batches (`solve_batch`, used by the angle sweeps, the agent polar tool and the ground-effect sweeps): the batches return `v_control` for each case, and `compute_loads_batch(v_controls=...)` uses it. Conditions: no leg forces, and no kernel cache on the lattice (with a kernel cache the linear lifting line builds its matrix from the cache and has no velocity tensor; only the vortex-lattice sweeps set a kernel cache). The nonlinear lifting line also takes its linear start solution from the same tensor. The vortex lattice keeps its own path (its force points are not its control points).

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

## Vortex-lattice pipelines

* **Repeated load points.** The loads evaluate the induced velocity at the force point and at the two leg mid-points of every panel. The leg mid-point of a panel is the leg mid-point of its neighbour, to the bit. With one core group (one surface, or surfaces joined into one group) the target core radius never acts, so `loads._load_points` gives `repeat = (keep, inverse)` and `induced_velocity(..., repeat=repeat)` runs the kernel on each point once. Each kernel row is computed alone, so the bits do not change. With the kernel cache, the cached part keeps all rows (its matrix-vector product can round differently when the rows change); only the wake legs use the points without repeats. With more than one core group there is no repeat (the target core radius depends on the strip width).
* **Sweeps.** `solve_sweep` of the vortex lattice out of ground effect sets the kernel cache and calls `solve_batch`: the systems are assembled one by one (as `assemble_vlm`, with the cache) and solved with one call of the dense solver (`core.solve_vlm_batch`); the loads of all angles are computed together. Each angle has the bits of the kernel-cache path that existed before. In ground effect the automatic chordwise count depends on the attitude: all angles use the count of the angle with the smallest gap to the ground and form one batch, so that the sweep contains no change of mesh. Each angle has the bits of a single solve with that count.
* **Probe lattices.** `resolve_n_chord`, `place_ground`, `find_bank_strike_limit` and `_fixed_n_chord` (in ground effect), `estimate_panels` and the mesh checks of the agent tools take their probe lattices from the lattice cache. The probe lattice has one chordwise panel and depends only on the aircraft geometry and the mesh settings. The results are identical because the probe lattices are read-only and identical to those built directly by `build_lattice`.
* **Small lattices in one batch.** `alpha_sweep` with `n_jobs="auto"` and the agent polar tool solve a vortex lattice of the size class "small" (`hardware.profile.size_class`) as one batch; a larger lattice keeps the pool of workers, which gains from cases in parallel (the dense solves of one process do not run in parallel). The agent polar batch has no kernel cache, so each angle equals `vt.analyze`.
* **Ground-effect CPU sweeps.** `GroundEffectSweep.run_sweep` solves its cases as batches on the CPU when `backend == "serial"` or when the lattice is of the size class "small" (`ventorum.hardware.profile.size_class`). It prepares all cases with one shared helper (`_prepare_sweep_cases`, also used by the GPU batch) and places the ground with the shared probe. One batch solves only cases with the same unknown map, so the valid cases are put into groups by their unknown map: the symmetric cases (bank angle 0, no sideslip) have about half the unknowns of the others. Each group is solved in chunks, and the chunk size comes from the unknown count of that group, so that the matrices of one chunk stay within about 256 MiB. The results go back in the order of the cases. The chunks use the CPU branch of the solver (`LatticeSolver.solve_batch` with the private flag `_cpu_only=True` and `continuation=False`), which skips the GPU check. The sweep does not change the device setting of `ventorum.gpu`, so sweeps in parallel threads do not change the device of other threads. The results are the same to round-off because each case uses the same lattice, the same ground plane and the same solver equations as a single solve, and continuation is disabled (the order and the grouping of the cases do not change a case).
* Lattice-only values cached on the lattice: the loads geometry (force points, leg mid-points and vectors), the load points and their repeats, the unknown map of the loads, the airfoil split (linear airfoils together), `has_tabulated`.
* The torch backend is not reproducible to the bit from one run to the next in ground effect (this was so before these changes); the parity tests leave it out of the ground cases.

## Small numerical rules

* `utils.vec.cross3` replaces `numpy.cross` on small arrays: the same formula and order of operations (the same bits), without the axis handling of `numpy.cross`.
* The Newton Jacobian of the nonlinear lifting line uses the scalar triple product `u_i . (v_ij x dl_i) = v_ij . (dl_i x u_i)`: one cross product per strip instead of one per pair.
* The symmetry test of the loads compares `|a - b| <= atol` directly (the same answer as `np.allclose` with `rtol = 0` for finite data).
* **Trefftz-plane half evaluation on the CPU.** `trefftz_induced_drag_batch` evaluates the normal wash only at the right strips in symmetric cases (`wake_dirs[k][1] == 0.0`, ground normal y-component zero, symmetric lattice, and symmetric circulation to relative 1e-12). It mirrors the normal wash to the left strips. The result is the same to round-off because physical symmetry makes the normal wash at each left strip equal to that of its mirror strip on the right.

## GPU pipelines

`ventorum/gpu` solves a batch of flight conditions on one lattice (or one large solve) on an NVIDIA CUDA GPU. The user guide is [Run solves on the GPU](../user/how_to_gpu). The modules:

| Module | Content |
| --- | --- |
| `gpu/__init__.py` | Device and precision settings (`set_device`, `set_precision`, `VENTORUM_DEVICE`, `VENTORUM_GPU_PRECISION`), availability, the `"auto"` rule (`use_gpu`: the cost model of the tuner, profile key `gpu`, `model`; without it the built-in `min_work` and `min_cases` per family). |
| `gpu/kernels.py` | NVIDIA Warp kernels, one set per precision: system matrices and velocity tensors (`tensors`, `normals`, the wake table), induced velocities for the loads, Trefftz normal wash, the nonlinear lifting-line state, trials and Jacobian, the float64 residual and contractions. |
| `gpu/engine.py` | GPU data of a lattice (`DeviceLattice`, cached in `geom_cache`), per-case data (`CaseData`), the batched solves, the loads, the polars on the GPU. |
| `gpu/pipeline.py` | Dispatch from `LatticeSolver.solve_batch` and `solve_lattice`; case setup and result objects with the CPU code. |

How a batch runs:

1. `solve_batch` (and `_solve_lattice` for one case) asks `gpu.pipeline.solve_batch` first. It returns None when the device is `"cpu"`, when `"auto"` estimates the CPU to be faster (`gpu.use_gpu(work, family, n_cases, n_panels)`: the tuned cost model of the machine, else the built-in work and case thresholds per family), inside a worker of a CPU pool with `"auto"`, or when the batch is not supported (Fourier solver; an airfoil type other than linear or tabulated; a nonlinear sweep with continuation and mixed kinds of cases). The CPU path then runs unchanged. Explicit ground planes (`solve_batch(..., grounds=...)`, used by the ground-effect sweeps) are supported.
2. The cases are split into groups with the same unknown map, ground presence and symmetry; each group is one GPU batch, and the results come back in the order of the cases.
3. The case setup (`_case_setup`: ground plane, validity checks, notes, wake direction) is the CPU code. The per-case inputs go to the GPU in one transfer (`engine.batch_data`); one kernel each makes the float64 wake-leg data of the float32 kernels (`case_te`, `case_points`), the Trefftz-plane geometry (`trefftz_prep`) and the terms of the lifting-line systems (`llt_rhs`). The polars of the strips are kept on the GPU and made again only when an airfoil changes (`engine.polar_struct`).
4. The system: the bound vortices and chordwise legs do not depend on the case, so their part is computed once per lattice and unknown map (`DeviceLattice.fixed`, the GPU kernel cache); each case adds its wake legs and ground images. On a lattice with more than one chordwise panel the wake legs come from a wake table: the panels of one strip with one core radius share their wake legs, so each case evaluates them once per wake group and point.
5. The dense solves (`engine.batched_solve`): small systems (`REF_MIN_N`) and small batches get a direct float64 LU. Otherwise one reference matrix per group of consecutive cases is inverted (float32), and each case is corrected with its residual in float64 until the relative residual is at the round-off level (float64 matrix) or below `REF_TOL32` (float32 matrix, three orders below the errors of its entries); a case that does not converge gets a direct solve.
6. The nonlinear lifting line runs Newton's method for all cases together (`engine.llt_nonlinear`). Per iteration: the Jacobian of the active cases (`nl_jacobian`, from the terms of `nl_state`), its LU in float32 (`newton_solve`: two refinements for a float64 Jacobian, none for a float32 one), one read of the velocity tensor for the Newton step and the fixed-point direction (`contract2`), the residuals of all trial points of the line search and of the fixed-point fallback (`nl_trials`, one thread per case, trial and strip; `nl_merit` sums them in strip order), the first trial that the CPU rules accept and the new state (`nl_update`, `nl_state`), and the merit and flags per case (`nl_reduce`). One transfer per iteration gives the active cases. Continuation and the CPU fallback are described in `gpu/pipeline.py` (`_nonlinear`).
7. The loads (`engine.loads`, two kernels: per strip and per case) are the equations of `compute_loads_batch` in float64 on the GPU. The vortex lattice takes the velocity at the load points from a cached matrix of the fixed legs (a product with the circulation, in float64), the wake legs per wake group, and the ground images per panel. The Trefftz plane evaluates the right strips only in symmetric cases.
8. One transfer brings all arrays to the host. `loads.loads_result` and `_case_result` make the result objects (the same functions as the CPU path); the statistics of the trust score come from the per-case kernel (the same rules as `trust.spanwise_stats_batch`).
9. The ground-effect sweeps (`ground_effect.sweep._gpu_batch`) check each case and place its ground as `analyze_ground_effect` does (`prepare_ground_case`), solve the valid cases as one batch with explicit ground planes, and make the results with `ground_case_result`. When the GPU does not take the batch, the CPU path batches small lattices and serial runs in chunks, and larger lattices keep the worker pool.

Rules for a change:

* The matrices are stored transposed (column-major per case, entry (i, j) at `[k, j, i]`): the threads of one warp then have the same sources and contiguous points. Keep this layout in new kernels.
* Warp kernels launch on the current PyTorch stream (`engine.launch`); a tensor viewed by a kernel must stay alive until the launch is queued (`CaseData` keeps the per-case tensors).
* float32 kernels must keep the three rules of the module notes of `gpu/kernels.py` (pair differences, float64 wake-leg data, float64 bound-vortex cross product). A new formula with a difference of nearly equal quantities needs the same care.
* A new per-case quantity of the loads must be added to `engine.loads` and to `loads.loads_result` (both paths make the objects there).
* Each GPU feature needs a parity test in `tests/test_gpu_pipeline.py` against the CPU path, in float64 (round-off) and float32.

## How to check a change

1. The verification report must not change (`validation/run_verification.py`).
2. The parity tests in `tests/test_kernels.py`, `tests/test_solve_overhead.py`, `tests/test_lattice_cache.py`, `tests/test_sweep_batch.py`, `tests/test_vlm_pipeline.py` and, on a machine with a GPU, `tests/test_gpu_pipeline.py`.
3. A repeated solve must give the same bits as the first solve (`test_repeated_solve_gives_the_same_bits`).
4. Timing: run on a quiet machine, with no agent or other heavy program, and compare the old and the new code side by side, alternating. A timing of the old code taken at another time is not a reference.
