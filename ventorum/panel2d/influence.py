# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Closed-form influences of two-dimensional panels.

Stream-function and velocity influences of linear-strength vortex panels
and constant-strength source panels, in the local frame of a panel: xi
along the panel from its start point, eta along the left normal (for a body
in Selig order, which is counterclockwise, the left normal points into the
body). The derivations are in ``docs/theory/panel2d.md``.

Conventions: a vortex sheet of strength gamma has the stream function
psi = (1/(4 pi)) int gamma(t) ln(r^2) dt (positive gamma turns clockwise);
a source sheet of strength sigma has psi = (1/(2 pi)) int sigma phi(t) dt,
with phi(t) the polar angle of the field point seen from the source point.
The source stream function is multivalued; each function below states the
branch cut that it uses.

References
----------
J. Katz and A. Plotkin, Low-Speed Aerodynamics, 2nd ed., Cambridge
University Press, 2001, chapters 10 and 11 (panel singularity elements).
M. Drela, XFOIL: An analysis and design system for low Reynolds number
airfoils, Lecture Notes in Engineering 54, Springer, 1989, pp. 1-12
(stream-function formulation with vortex and source panels).
"""

from __future__ import annotations

import math

import numpy as np
import torch

__all__ = [
    "linear_vortex_stream_influence",
    "source_panel_stream_influence",
    "wake_source_stream_influence",
    "image_source_stream_influence",
    "linear_vortex_velocity_influence",
    "source_panel_velocity_influence",
    "compute_panel_local_coordinates_numpy",
    "compute_panel_local_coordinates_torch",
]

_TWO_PI = 2.0 * math.pi


def _xp_of(*args):
    """Return torch when any argument is a tensor, else numpy."""
    for a in args:
        if isinstance(a, torch.Tensor):
            return torch
    return np


def _as(xp, x, like=None):
    """Return ``x`` as a floating-point array of the ``xp`` module."""
    if xp is np:
        return np.asarray(x, dtype=np.float64)
    if isinstance(x, torch.Tensor):
        return x
    if like is not None:
        return torch.as_tensor(x, dtype=like.dtype, device=like.device)
    return torch.as_tensor(x, dtype=torch.float64)


def _scalar_out(out, *inputs):
    """Return floats when every input is a scalar."""
    if all(np.ndim(a) == 0 and not isinstance(a, torch.Tensor) for a in inputs):
        if isinstance(out, tuple):
            return tuple(float(o) for o in out)
        return float(out)
    return out


def _tiny(xp, a):
    """Return a tiny positive number of the precision of ``a``."""
    if xp is np:
        return 1e-300
    return torch.finfo(a.dtype).tiny


def _clean_eta(xp, eta, length):
    """Return eta with round-off values set to +0 (the inner side of a body panel)."""
    return xp.where(abs(eta) < 1e-12 * length, 0.0 * eta + 0.0, eta)


def _f0(xp, t, eta):
    """Return the antiderivative of ln(t^2 + eta^2) with respect to t."""
    tiny = _tiny(xp, t)
    d2 = t * t + eta * eta
    safe = xp.where(d2 > tiny, d2, tiny + 0.0 * d2)
    small = abs(eta) < tiny
    seta = xp.where(small, 1.0 + 0.0 * eta, eta)
    return t * xp.log(safe) - 2.0 * t + xp.where(small, 0.0 * eta, 2.0 * eta * xp.arctan(t / seta))


def _f1(xp, t, eta):
    """Return the antiderivative of t ln(t^2 + eta^2) with respect to t."""
    tiny = _tiny(xp, t)
    d2 = t * t + eta * eta
    safe = xp.where(d2 > tiny, d2, tiny + 0.0 * d2)
    return 0.5 * d2 * (xp.log(safe) - 1.0)


def _far_terms(xp, x, e, L):
    """Return (P0, P1, I0, Lam) of a panel in the form that is stable far from it.

    With xc = xi - L/2 (field point relative to the panel middle),
    a = xi, b = xi - L, ra^2 = a^2 + eta^2, rb^2 = b^2 + eta^2::

        Lam = ln(ra^2 / rb^2) = log1p(2 xc L / rb^2)
        I0  = int_0^L eta / r^2 dt = atan2(eta L, eta^2 + a b)
        P0  = int_0^L ln r^2 dt = L ln(ra rb) + xc Lam - 2 L + 2 eta I0
        P1  = int_0^L (t - L/2) ln r^2 dt = (Lam/2) (xc^2 - L^2/4 - eta^2) - xc L + 2 xc eta I0

    The terms of P0 do not cancel for distant points, so the form keeps the
    relative precision of float32 arithmetic.
    """
    tiny = _tiny(xp, x)
    xc = x - 0.5 * L
    b = x - L
    ra2 = x * x + e * e
    rb2 = b * b + e * e
    rb2 = xp.where(rb2 > tiny, rb2, tiny + 0.0 * rb2)
    ra2 = xp.where(ra2 > tiny, ra2, tiny + 0.0 * ra2)
    arg = 2.0 * xc * L / rb2
    lam = xp.log1p(xp.where(arg > -1.0, arg, 0.0 * arg))
    i0 = xp.arctan2(e * L, e * e + x * b)
    p0 = 0.5 * L * (xp.log(ra2) + xp.log(rb2)) + xc * lam - 2.0 * L + 2.0 * e * i0
    p1 = 0.5 * lam * (xc * xc - 0.25 * L * L - e * e) - xc * L + 2.0 * xc * e * i0
    return p0, p1, i0, lam


def _far(x, e, L):
    """Return a mask of field points farther than two panel lengths from the panel middle."""
    xc = x - 0.5 * L
    return xc * xc + e * e > 4.0 * L * L


def linear_vortex_stream_influence(xi, eta, length):
    """Return the stream-function influence of a linear-strength vortex panel.

    The vortex strength varies linearly from gamma_1 at xi = 0 to gamma_2 at
    xi = length: gamma(t) = gamma_1 (1 - t/L) + gamma_2 t/L. Near the panel
    the antiderivatives at the panel ends are used; farther than two panel
    lengths from the panel middle the equivalent form of :func:`_far_terms`
    is used, which avoids the cancellation of the end terms (both forms are
    exact).

    Parameters
    ----------
    xi, eta : float or array
        Local coordinates of the field point [m].
    length : float or array
        Panel length [m].

    Returns
    -------
    tuple
        ``(psi_1, psi_2)``: stream function [m] per unit gamma_1 and
        gamma_2 (gamma over the reference speed).
    """
    xp = _xp_of(xi, eta, length)
    x = _as(xp, xi)
    e = _as(xp, eta, x if xp is torch else None)
    L = _as(xp, length, x if xp is torch else None)
    k = 1.0 / (4.0 * math.pi)
    i0 = _f0(xp, x, e) - _f0(xp, x - L, e)
    df1 = _f1(xp, x, e) - _f1(xp, x - L, e)
    near1 = k * ((1.0 - x / L) * i0 + df1 / L)
    near2 = k * ((x / L) * i0 - df1 / L)
    p0, p1, _, _ = _far_terms(xp, x, e, L)
    far1 = k * (0.5 * p0 - p1 / L)
    far2 = k * (0.5 * p0 + p1 / L)
    far = _far(x, e, L)
    psi1 = xp.where(far, far1, near1)
    psi2 = xp.where(far, far2, near2)
    return _scalar_out((psi1, psi2), xi, eta, length)


def _source_standard(xp, x, e, L):
    """Return int_0^L atan2(eta, xi - t) dt on the branch (-pi, pi].

    Near the panel the antiderivative u atan2(eta, u) + (eta/2) ln(u^2 +
    eta^2) at the two ends is used; far from it the equivalent form
    L (phi_a + phi_b)/2 - xc I0 + eta Lam / 2 (see :func:`_far_terms`).
    """
    t1 = x
    t2 = x - L
    r1 = t1 * t1 + e * e
    r2 = t2 * t2 + e * e
    tiny = _tiny(xp, x)
    r1 = xp.where(r1 > tiny, r1, tiny + 0.0 * r1)
    r2 = xp.where(r2 > tiny, r2, tiny + 0.0 * r2)
    pa = xp.arctan2(e, t1)
    pb = xp.arctan2(e, t2)
    near = t1 * pa + 0.5 * e * xp.log(r1) - (t2 * pb + 0.5 * e * xp.log(r2))
    _, _, i0, lam = _far_terms(xp, x, e, L)
    far = 0.5 * L * (pa + pb) - (x - 0.5 * L) * i0 + 0.5 * e * lam
    return xp.where(_far(x, e, L), far, near)


def source_panel_stream_influence(xi, eta, length, theta_panel=0.0):
    """Return the stream-function influence of a constant-strength body source panel.

    The polar angle phi of the field point seen from a source point has its
    branch cut along the local -eta direction (for a body panel in Selig
    order this is the outward normal), so phi lies in (-pi/2, 3pi/2]. A
    field point with eta = 0 (or a round-off value) is taken on the +eta
    side, which is the inner side of a body panel. With this branch the
    stream function is continuous over the body interior and over the whole
    plane except the strip {eta < 0, 0 < xi < L} on the outer side of the
    panel; a field point must not lie in that strip (this holds for the
    nodes of a single airfoil).

    Parameters
    ----------
    xi, eta : float or array
        Local coordinates of the field point [m].
    length : float or array
        Panel length [m].
    theta_panel : float or array, optional
        Panel angle to the global x axis [rad]. It adds the constant
        theta_panel L / (2 pi), which converts the local angle to the
        global angle. Default 0.

    Returns
    -------
    float or array
        Stream function [m] per unit source strength (over the reference
        speed).
    """
    xp = _xp_of(xi, eta, length, theta_panel)
    x = _as(xp, xi)
    L = _as(xp, length, x if xp is torch else None)
    e = _clean_eta(xp, _as(xp, eta, x if xp is torch else None), L)
    th = _as(xp, theta_panel, x if xp is torch else None)
    val = _source_standard(xp, x, e, L)
    xc = xp.where(x < 0.0, 0.0 * x, xp.where(x > L, L + 0.0 * x, x))
    corr = xp.where(e < 0.0, L - xc, 0.0 * x)
    out = val / _TWO_PI + corr + th * L / _TWO_PI
    return _scalar_out(out, xi, eta, length)


def wake_source_stream_influence(xi, eta, length):
    """Return the stream-function influence of a constant-strength wake source panel.

    The polar angle has its branch cut along the local +xi direction
    (downstream along a wake panel), so phi lies in [0, 2 pi). The stream
    function is then continuous everywhere upstream of the wake, also over
    the airfoil.

    Parameters
    ----------
    xi, eta : float or array
        Local coordinates of the field point [m].
    length : float or array
        Panel length [m].

    Returns
    -------
    float or array
        Stream function [m] per unit source strength.
    """
    xp = _xp_of(xi, eta, length)
    x = _as(xp, xi)
    L = _as(xp, length, x if xp is torch else None)
    e = _clean_eta(xp, _as(xp, eta, x if xp is torch else None), L)
    val = _source_standard(xp, x, e, L)
    out = val / _TWO_PI + xp.where(e < 0.0, L + 0.0 * x, 0.0 * x)
    return _scalar_out(out, xi, eta, length)


def image_source_stream_influence(field, p_start, p_end):
    """Return the stream-function influence of source panels below the ground.

    For a field point above the ground (y > 0) and a source panel below it
    (y < 0), the global polar angle lies in (0, pi) for every source point;
    this branch (cut along -y) is used. Images of ground-effect bodies and
    wakes are such panels.

    Parameters
    ----------
    field : array
        Field points, shape (..., Nc, 2) [m].
    p_start, p_end : array
        Panel end points, shape (..., Np, 2) [m].

    Returns
    -------
    array
        Stream function [m] per unit source strength, shape (..., Nc, Np).
    """
    xp = _xp_of(field, p_start, p_end)
    if xp is np:
        xi, eta, L, th = compute_panel_local_coordinates_numpy(field, p_start, p_end)
    else:
        xi, eta, L, th = compute_panel_local_coordinates_torch(field, p_start, p_end)
    val = _source_standard(xp, xi, eta, L)
    dx = field[..., :, None, 0] - p_start[..., None, :, 0]
    dy = field[..., :, None, 1] - p_start[..., None, :, 1]
    phi_g = xp.arctan2(dy, dx)
    phi_l = xp.arctan2(eta, xi)
    k = xp.round((phi_g - phi_l - th) / _TWO_PI)
    return (val + (th + _TWO_PI * k) * L) / _TWO_PI


def linear_vortex_velocity_influence(xi, eta, length):
    """Return the velocity influence of a linear-strength vortex panel.

    Components in the local frame (along xi and eta) per unit nodal
    strength gamma_1 (at xi = 0) and gamma_2 (at xi = L)::

        u = (1/(2 pi)) int gamma(t) eta / r^2 dt
        v = -(1/(2 pi)) int gamma(t) (xi - t) / r^2 dt

    Parameters
    ----------
    xi, eta : float or array
        Local coordinates of the field point [m] (off the panel).
    length : float or array
        Panel length [m].

    Returns
    -------
    tuple
        ``(u1, v1, u2, v2)`` [-] per unit gamma_1 and gamma_2.
    """
    xp = _xp_of(xi, eta, length)
    x = _as(xp, xi)
    e = _as(xp, eta, x if xp is torch else None)
    L = _as(xp, length, x if xp is torch else None)
    t2 = x - L
    r1 = x * x + e * e
    r2 = t2 * t2 + e * e
    i0 = xp.arctan2(e, t2) - xp.arctan2(e, x)
    j0 = 0.5 * xp.log(r1 / r2)
    i1 = x * i0 - e * j0
    j1 = x * j0 - L + e * i0
    k = 1.0 / _TWO_PI
    u1 = k * (i0 - i1 / L)
    u2 = k * (i1 / L)
    v1 = -k * (j0 - j1 / L)
    v2 = -k * (j1 / L)
    return _scalar_out((u1, v1, u2, v2), xi, eta, length)


def source_panel_velocity_influence(xi, eta, length):
    """Return the velocity influence of a constant-strength source panel.

    Components in the local frame per unit source strength::

        u = (1/(4 pi)) ln(r1^2 / r2^2),   v = (1/(2 pi)) (atan2(eta, xi - L) - atan2(eta, xi))

    with r1 and r2 the distances to the panel ends. On the panel itself
    (eta = 0, 0 < xi < L) the normal component is the value on the +eta
    side.

    Parameters
    ----------
    xi, eta : float or array
        Local coordinates of the field point [m] (not at a panel end).
    length : float or array
        Panel length [m].

    Returns
    -------
    tuple
        ``(u, v)`` [-] per unit source strength.
    """
    xp = _xp_of(xi, eta, length)
    x = _as(xp, xi)
    L = _as(xp, length, x if xp is torch else None)
    e = _clean_eta(xp, _as(xp, eta, x if xp is torch else None), L)
    t2 = x - L
    r1 = x * x + e * e
    r2 = t2 * t2 + e * e
    u = xp.log(r1 / r2) / (4.0 * math.pi)
    v = (xp.arctan2(e, t2) - xp.arctan2(e, x)) / _TWO_PI
    return _scalar_out((u, v), xi, eta, length)


def compute_panel_local_coordinates_numpy(colloc, p_start, p_end):
    """Transform field points into the local frames of panels (numpy).

    Parameters
    ----------
    colloc : np.ndarray
        Field points, shape (Nc, 2) or (..., Nc, 2) [m].
    p_start, p_end : np.ndarray
        Panel start and end points, shape (Np, 2) or (..., Np, 2) [m].

    Returns
    -------
    tuple
        ``(xi, eta, length, theta_panel)``: xi and eta of shape (..., Nc,
        Np) [m], length and theta_panel of shape (..., 1, Np) [m], [rad].
    """
    dx = p_end[..., :, 0] - p_start[..., :, 0]
    dy = p_end[..., :, 1] - p_start[..., :, 1]
    length = np.hypot(dx, dy)
    tx = dx / length
    ty = dy / length
    theta_panel = np.arctan2(dy, dx)
    rx = colloc[..., :, None, 0] - p_start[..., None, :, 0]
    ry = colloc[..., :, None, 1] - p_start[..., None, :, 1]
    xi = rx * tx[..., None, :] + ry * ty[..., None, :]
    eta = -rx * ty[..., None, :] + ry * tx[..., None, :]
    return xi, eta, length[..., None, :], theta_panel[..., None, :]


def compute_panel_local_coordinates_torch(colloc, p_start, p_end):
    """Transform field points into the local frames of panels (torch).

    Parameters
    ----------
    colloc : torch.Tensor
        Field points, shape (..., Nc, 2) [m].
    p_start, p_end : torch.Tensor
        Panel start and end points, shape (..., Np, 2) [m].

    Returns
    -------
    tuple
        ``(xi, eta, length, theta_panel)``: xi and eta of shape (..., Nc,
        Np) [m], length and theta_panel of shape (..., 1, Np) [m], [rad].
    """
    dx = p_end[..., :, 0] - p_start[..., :, 0]
    dy = p_end[..., :, 1] - p_start[..., :, 1]
    length = torch.hypot(dx, dy)
    tx = dx / length
    ty = dy / length
    theta_panel = torch.atan2(dy, dx)
    rx = colloc[..., :, None, 0] - p_start[..., None, :, 0]
    ry = colloc[..., :, None, 1] - p_start[..., None, :, 1]
    xi = rx * tx[..., None, :] + ry * ty[..., None, :]
    eta = -rx * ty[..., None, :] + ry * tx[..., None, :]
    return xi, eta, length[..., None, :], theta_panel[..., None, :]
