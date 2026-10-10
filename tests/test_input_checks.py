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
