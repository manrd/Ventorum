# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Forward-mode first derivatives for the boundary-layer equations.

A :class:`Dual` holds an array of values and the array of their first
derivatives with respect to a fixed list of independent variables. The
arithmetic below applies the chain rule. The closure functions of
:mod:`ventorum.ibl2d.closure` return analytic partial derivatives; the
function :func:`lift` combines them with the derivatives of their inputs.
The result is the exact Jacobian of the discrete equations (to round-off),
which the Newton solvers need. The same code runs on numpy arrays and on
torch tensors.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Dual", "variables", "lift", "where", "constant"]


class Dual:
    """Values with first derivatives.

    Parameters
    ----------
    val : array
        Values, shape ``S``.
    der : array
        Derivatives with respect to ``n`` independent variables, shape
        ``S + (n,)``.
    xp : module
        Array module (numpy or torch).
    """

    __slots__ = ("val", "der", "xp")
    # Make numpy arrays defer to the reflected operators of this class.
    __array_ufunc__ = None

    def __init__(self, val, der, xp):
        self.val = val
        self.der = der
        self.xp = xp

    def _wrap(self, other):
        if isinstance(other, Dual):
            return other
        return constant(other, self)

    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val + other.val, self.der + other.der, self.xp)
        return Dual(self.val + other, self.der, self.xp)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val - other.val, self.der - other.der, self.xp)
        return Dual(self.val - other, self.der, self.xp)

    def __rsub__(self, other):
        return Dual(other - self.val, -self.der, self.xp)

    def __neg__(self):
        return Dual(-self.val, -self.der, self.xp)

    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(
                self.val * other.val,
                self.der * other.val[..., None] + other.der * self.val[..., None],
                self.xp,
            )
        o = other
        if not np.isscalar(o):
            o = o[..., None]
        return Dual(self.val * other, self.der * o, self.xp)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Dual):
            inv = 1.0 / other.val
            v = self.val * inv
            return Dual(v, (self.der - other.der * v[..., None]) * inv[..., None], self.xp)
        o = other
        if not np.isscalar(o):
            o = o[..., None]
        return Dual(self.val / other, self.der / o, self.xp)

    def __rtruediv__(self, other):
        inv = 1.0 / self.val
        v = other * inv
        return Dual(v, -self.der * (v * inv)[..., None], self.xp)

    def __pow__(self, p):
        v = self.val**p
        return Dual(v, self.der * (p * self.val ** (p - 1.0))[..., None], self.xp)

    def log(self):
        """Return the natural logarithm."""
        return Dual(self.xp.log(self.val), self.der / self.val[..., None], self.xp)

    def exp(self):
        """Return the exponential."""
        v = self.xp.exp(self.val)
        return Dual(v, self.der * v[..., None], self.xp)

    def sqrt(self):
        """Return the square root."""
        v = self.xp.sqrt(self.val)
        return Dual(v, self.der * (0.5 / v)[..., None], self.xp)

    def __getitem__(self, idx):
        return Dual(self.val[idx], self.der[idx], self.xp)


def constant(value, like: Dual) -> Dual:
    """Return a Dual with zero derivatives and the shape of ``like``.

    Parameters
    ----------
    value : float or array
        Value.
    like : Dual
        Template for the shape and the number of variables.

    Returns
    -------
    Dual
        The constant.
    """
    v = like.val * 0.0 + value
    return Dual(v, like.der * 0.0, like.xp)


def variables(values, xp=None):
    """Return independent variables with unit derivative seeds.

    Parameters
    ----------
    values : list of array
        The ``n`` independent variables, each of shape ``S``.
    xp : module, optional
        Array module. Default numpy.

    Returns
    -------
    list of Dual
        Variable ``k`` has the derivative ``e_k``.
    """
    xp = np if xp is None else xp
    n = len(values)
    out = []
    for k, v in enumerate(values):
        if xp is np:
            der = np.zeros(np.shape(v) + (n,), dtype=np.result_type(v, np.float64))
        else:
            der = xp.zeros(tuple(v.shape) + (n,), dtype=v.dtype, device=v.device)
        der[..., k] = 1.0
        out.append(Dual(v, der, xp))
    return out


def lift(result, inputs) -> Dual:
    """Combine a closure value and its partial derivatives with the chain rule.

    Parameters
    ----------
    result : tuple
        ``(value, d_value_d_input_0, d_value_d_input_1, ...)`` as returned
        by a closure function.
    inputs : list of Dual
        The inputs of the closure, in the order of the partial derivatives.

    Returns
    -------
    Dual
        The closure value with its derivatives.
    """
    value = result[0]
    der = None
    for part, inp in zip(result[1:], inputs):
        term = inp.der * part[..., None]
        der = term if der is None else der + term
    return Dual(value, der, inputs[0].xp)


def where(cond, a: Dual, b: Dual) -> Dual:
    """Select elementwise between two Duals.

    Parameters
    ----------
    cond : array of bool
        Selector, shape ``S``.
    a, b : Dual
        Values where ``cond`` is true and false.

    Returns
    -------
    Dual
        The selection.
    """
    xp = a.xp
    return Dual(xp.where(cond, a.val, b.val), xp.where(cond[..., None], a.der, b.der), xp)
