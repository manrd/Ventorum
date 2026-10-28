"""Input checks for non-finite and extreme inputs.

Every input that cannot give a valid result is refused before the solve,
with a ``ValueError`` that names the field and the value.
"""

import numpy as np
import pytest

import ventorum as vt
from ventorum.core.trust import evaluate_aerodynamic_trust
from ventorum.utils import validation as V

BAD_VALUES = [float("nan"), float("inf"), float("-inf")]


def _wing(**kw):
    sections = [
        vt.WingSection(y_frac=0.0, chord=1.5),
        vt.WingSection(y_frac=1.0, chord=1.0),
    ]
    params = {"name": "Wing", "semi_span": 5.0, "sections": sections}
    params.update(kw)
    return vt.LiftingSurface(**params)


def _condition(**kw):
    params = {"V_inf": 40.0, "alpha": np.radians(5.0), "rho": 1.225}
    params.update(kw)
    return vt.FlightCondition(**params)


def _settings(**kw):
    params = {"n_panels": 12}
    params.update(kw)
    return vt.SolverSettings(**params)


def _check_field(field, bad):
    """Mutate one field to *bad* and check the validator refuses it."""
    if field in ("V_inf", "alpha", "rho", "h"):
        with pytest.raises(ValueError) as exc:
            V.validate_flight_condition(_condition(**{field: bad}))
    elif field == "chord":
        wing = _wing()
        wing.sections[0].chord = bad
        with pytest.raises(ValueError) as exc:
            V.validate_surface(wing)
    elif field == "semi_span":
        wing = _wing()
        wing.semi_span = bad
        with pytest.raises(ValueError) as exc:
            V.validate_surface(wing)
    elif field == "x_le":
        wing = _wing()
        wing.sections[0].x_le = bad
        with pytest.raises(ValueError) as exc:
            V.validate_surface(wing)
    elif field == "position":
        wing = _wing()
        wing.position = np.array([bad, 0.0, 0.0])
        with pytest.raises(ValueError) as exc:
            V.validate_surface(wing)
    else:  # pragma: no cover
        raise AssertionError(f"Unknown field {field!r}.")
    assert field in str(exc.value)


@pytest.mark.parametrize("bad", BAD_VALUES)
@pytest.mark.parametrize(
    "field", ["V_inf", "alpha", "rho", "h", "chord", "semi_span", "x_le", "position"]
)
def test_nan_and_inf_inputs_are_refused(field, bad):
    _check_field(field, bad)


def test_nan_condition_is_refused_by_analyze():
    """The reproduce case of the 4th review: NaN speed gives an error, not NaN results."""
    with pytest.raises(ValueError) as exc:
        vt.analyze(_wing(), condition=vt.FlightCondition(V_inf=float("nan"), alpha=0.07))
    assert "V_inf" in str(exc.value)


def test_nan_height_message_names_the_height():
    with pytest.raises(ValueError) as exc:
        V.validate_flight_condition(_condition(h=float("nan")))
    assert "finite" in str(exc.value).lower()


def test_absurd_sizes_are_refused():
    for bad in (1e300, 2e5):
        wing = _wing()
        wing.position = np.array([bad, 0.0, 0.0])
        with pytest.raises(ValueError) as exc:
            V.validate_surface(wing)
        assert "position" in str(exc.value)
    wing = _wing()
    wing.position = np.array([9e4, 0.0, 0.0])
    V.validate_surface(wing)


def test_numpy_integers_are_accepted():
    surf = _wing(n_panels=np.int64(24))
    V.validate_surface(surf)
    assert surf.n_panels == 24 and isinstance(surf.n_panels, int)
    sett = _settings(n_panels=np.int64(24))
    V.validate_solver_settings(sett)
    assert sett.n_panels == 24 and isinstance(sett.n_panels, int)
    ref = vt.analyze(_wing(), alpha_deg=5.0, settings=_settings(n_panels=24))
    got = vt.analyze(_wing(), alpha_deg=5.0, settings=_settings(n_panels=np.int64(24)))
    assert got.totals.CL == pytest.approx(ref.totals.CL)
    ref = vt.analyze(_wing(n_panels=24), alpha_deg=5.0, settings=_settings())
    got = vt.analyze(_wing(n_panels=np.int64(24)), alpha_deg=5.0, settings=_settings())
    assert got.totals.CL == pytest.approx(ref.totals.CL)
    with pytest.raises(ValueError) as exc:
        V.validate_solver_settings(_settings(n_panels=True))
    assert "n_panels" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        V.validate_surface(_wing(n_panels=True))
    assert "n_panels" in str(exc.value)


def test_identical_surfaces_are_refused():
    first = _wing()
    second = first.clone()
    aircraft = vt.Aircraft(name="Twin", surfaces=[first, second])
    with pytest.raises(ValueError) as exc:
        vt.analyze(aircraft, alpha_deg=5.0, settings=_settings())
    assert "Wing" in str(exc.value)


def test_overlapping_mirrored_surfaces_are_refused():
    right = vt.LiftingSurface(
        name="Right",
        semi_span=5.0,
        is_symmetric=False,
        position=np.array([0.0, -2.5, 0.0]),
        spacing="uniform",
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.5),
            vt.WingSection(y_frac=1.0, chord=1.5),
        ],
    )
    left = right.mirrored("Left")
    aircraft = vt.Aircraft(name="Overlap", surfaces=[right, left])
    with pytest.raises(ValueError) as exc:
        vt.analyze(aircraft, alpha_deg=5.0, settings=_settings())
    msg = str(exc.value)
    assert "Right" in msg and "Left" in msg


def test_shared_edge_surfaces_still_pass():
    one = _wing()
    right = _wing(name="RightHalf", is_symmetric=False)
    left = right.mirrored("LeftHalf")
    split = vt.Aircraft(name="Split", surfaces=[right, left])
    ref = vt.analyze(one, alpha_deg=5.0, settings=_settings())
    got = vt.analyze(split, alpha_deg=5.0, settings=_settings())
    assert got.totals.CL == pytest.approx(ref.totals.CL, rel=0.01)


@pytest.mark.parametrize("bad", BAD_VALUES)
def test_non_finite_result_is_never_high_trust(bad):
    trust = evaluate_aerodynamic_trust(AR=8.0, CL=bad, CDi=0.01, max_sweep_rad=0.0)
    assert trust.rating in ("LOW", "UNRELIABLE")     # never above LOW (card, decision 6)
    assert any("non-finite result" in w.lower() for w in trust.warnings)


def test_non_finite_guard_never_raises_the_trust_level():
    """The guard caps the level at LOW; it must not lift UNRELIABLE to LOW."""
    import numpy as np

    from ventorum.core.trust import evaluate_aerodynamic_trust

    kw = dict(AR=0.5, CL=0.5, CDi=0.01, converged=False, n_panels=4, max_sweep_rad=np.radians(60.0),
              solver_type="linear")
    assert evaluate_aerodynamic_trust(**kw).rating == "UNRELIABLE"
    kw["CL"] = float("nan")
    t = evaluate_aerodynamic_trust(**kw)
    assert t.rating == "UNRELIABLE"
    assert any("non-finite result" in w.lower() for w in t.warnings)


def test_tiny_span_chord_and_area_are_refused():
    """validate_surface has lower limits on span, chord and area."""
    wing = _wing()
    wing.semi_span = 1e-300
    with pytest.raises(ValueError) as exc:
        V.validate_surface(wing)
    assert "semi_span" in str(exc.value)
    wing = _wing()
    wing.sections[0].chord = 1e-300
    wing.sections[1].chord = 1e-300
    with pytest.raises(ValueError) as exc:
        V.validate_surface(wing)
    assert "chord" in str(exc.value)
    for bad in (0.0, -1.0):
        wing = _wing()
        wing.semi_span = bad
        with pytest.raises(ValueError) as exc:
            V.validate_surface(wing)
        assert "semi_span" in str(exc.value)


def test_small_but_sane_wing_still_passes():
    """The lower limits do not refuse a small (1 cm) wing."""
    wing = _wing(semi_span=0.005, sections=[
        vt.WingSection(y_frac=0.0, chord=0.01),
        vt.WingSection(y_frac=1.0, chord=0.01),
    ])
    V.validate_surface(wing)


def test_zero_reference_values_are_refused():
    """A zero S_ref, b_ref or c_ref is refused."""
    for key in ("S_ref", "b_ref", "c_ref"):
        ac = vt.Aircraft(name="A", surfaces=[_wing()], **{key: 0.0})
        with pytest.raises(ValueError) as exc:
            V.validate_aircraft(ac)
        assert key in str(exc.value)


def test_tiny_reference_values_are_refused():
    """S_ref, b_ref and c_ref below their lower limits are refused; the limits pass.

    Before the fix, b_ref = 1e-300 gave a ZeroDivisionError and
    S_ref = 1e-300 an OverflowError in the loads.
    """
    floors = {"S_ref": V.MIN_AREA_M2, "b_ref": V.MIN_SPAN_M, "c_ref": V.MIN_CHORD_M}
    for key, floor in floors.items():
        for bad in (1e-300, 0.5 * floor):
            ac = vt.Aircraft(name="A", surfaces=[_wing()], **{key: bad})
            with pytest.raises(ValueError) as exc:
                V.validate_aircraft(ac)
            assert key in str(exc.value)
        V.validate_aircraft(vt.Aircraft(name="A", surfaces=[_wing()], **{key: floor}))


def test_speed_below_minimum_is_refused():
    """V_inf below 0.1 m/s is refused; 0.1 m/s is accepted.

    Before the fix, V_inf = 1e-300 passed the check and gave a
    ZeroDivisionError in the loads.
    """
    assert V.MIN_SPEED_M_S == 0.1
    for bad in (1e-300, 0.05):
        with pytest.raises(ValueError) as exc:
            V.validate_flight_condition(vt.FlightCondition(V_inf=bad))
        assert "V_inf" in str(exc.value)
    V.validate_flight_condition(vt.FlightCondition(V_inf=0.1))
    with pytest.raises(ValueError):
        vt.analyze(_wing(), V_inf=1e-300, n_panels=8)


def test_ground_effect_default_alpha_is_five_deg():
    """The ground-effect Python API uses alpha = 5 deg when alpha_deg is omitted."""
    import dataclasses
    import inspect

    from ventorum.ground_effect import GroundEffectCondition, analyze_ground_effect, sweep_height, sweep_roll
    from ventorum.ground_effect.solver import prepare_ground_case

    for fn in (analyze_ground_effect, prepare_ground_case, sweep_height, sweep_roll):
        assert inspect.signature(fn).parameters["alpha_deg"].default == 5.0, fn.__name__
    # prepare_ground_case refers to the parameters of analyze_ground_effect.
    for fn in (analyze_ground_effect, sweep_height, sweep_roll, GroundEffectCondition):
        assert "(default 5.0)" in " ".join(fn.__doc__.split()), fn.__name__
    fields = {f.name: f for f in dataclasses.fields(GroundEffectCondition)}
    assert fields["alpha_deg"].default == 5.0
    assert GroundEffectCondition().alpha_deg == 5.0
    from ventorum.ground_effect import GroundEffectSweep
    assert tuple(inspect.signature(GroundEffectSweep.run_sweep).parameters["alphas_deg"].default) == (5.0,)
    assert "(default 5.0)" in " ".join(GroundEffectSweep.run_sweep.__doc__.split())


def test_mach_warning_uses_the_isa_speed_of_sound():
    """The Mach warning of the input check uses the same speed of sound as the trust score."""
    import warnings

    from ventorum.core.trust import isa_speed_of_sound

    rho = 0.4   # about 7 km in the ISA atmosphere
    v = 100.0   # Mach 0.29 at sea level, Mach 0.34 at rho = 0.4 kg/m^3
    assert v / isa_speed_of_sound(rho) > 0.3 > v / isa_speed_of_sound(1.225)
    with pytest.warns(RuntimeWarning, match=r"Mach 0\.34 .*ISA speed of sound"):
        V.validate_flight_condition(vt.FlightCondition(V_inf=v, rho=rho))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        V.validate_flight_condition(vt.FlightCondition(V_inf=v, rho=1.225))


def _half_wing_and_mirror(spacing="cosine"):
    right = vt.LiftingSurface(
        name="RightHalf",
        semi_span=4.0,
        is_symmetric=False,
        spacing=spacing,
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    return vt.Aircraft(name="Split", surfaces=[right, right.mirrored("LeftHalf")])


def _wing_with_winglets():
    wing = vt.LiftingSurface(
        name="Wing",
        semi_span=4.0,
        spacing="cosine",
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=1.0),
        ],
    )
    winglet = vt.LiftingSurface(
        name="Winglet",
        semi_span=0.8,
        dihedral=np.radians(90.0),
        position=np.array([0.0, 4.0, 0.0]),
        is_symmetric=False,
        spacing="cosine",
        sections=[
            vt.WingSection(y_frac=0.0, chord=1.0),
            vt.WingSection(y_frac=1.0, chord=0.5),
        ],
    )
    return vt.Aircraft(name="Winglets", surfaces=[wing, winglet, winglet.mirrored()])


def test_joined_halves_pass_on_a_fine_cosine_mesh():
    """A half wing from y = 0 plus its mirror copy shares the root edge only.

    Regression: with cosine spacing at 80 strips the root strips are narrower
    than 1.0e-3 chords, and the overlap check refused the valid join.
    """
    aircraft = _half_wing_and_mirror()
    settings = _settings(n_panels=80, n_chord=4, solver_type="vlm")
    got = vt.analyze(aircraft, alpha_deg=4.0, settings=settings)
    coarse = vt.analyze(aircraft, alpha_deg=4.0, settings=_settings(n_panels=40, n_chord=4, solver_type="vlm"))
    assert got.totals.CL == pytest.approx(coarse.totals.CL, rel=1e-3)


def test_wing_with_joined_winglets_passes_on_a_fine_cosine_mesh():
    """A wing plus 90 deg winglets on separate joined surfaces is valid at 60 strips and more."""
    aircraft = _wing_with_winglets()
    for n in (60, 80):
        res = vt.analyze(aircraft, alpha_deg=4.0, settings=_settings(n_panels=n, n_chord=4, solver_type="vlm"))
        assert np.isfinite(res.totals.CL)


@pytest.mark.parametrize("dz", [0.0, 1.0e-4])
def test_stacked_surfaces_are_refused_on_a_fine_cosine_mesh(dz):
    """Two surfaces that lie on each other stay refused on a fine clustered mesh."""
    aircraft = _half_wing_and_mirror()
    left = aircraft.surfaces[1]
    copy = left.clone()
    copy.name = "Copy"
    copy.position = np.asarray(copy.position, dtype=float) + np.array([0.0, 0.0, dz])
    stacked = vt.Aircraft(name="Stacked", surfaces=[aircraft.surfaces[0], left, copy])
    with pytest.raises(ValueError) as exc:
        vt.analyze(stacked, alpha_deg=4.0, settings=_settings(n_panels=80, n_chord=4, solver_type="vlm"))
    msg = str(exc.value)
    assert "overlap" in msg and "Copy" in msg
