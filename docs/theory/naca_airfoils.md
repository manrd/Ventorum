---
orphan: true
---

# NACA 4-digit and 5-digit airfoils

This chapter gives the thickness distribution and the mean lines that
`ventorum.airfoil.naca` builds into unit-chord coordinates. All quantities
are fractions of chord [-] unless the text says otherwise. The coordinates
use Selig order: from the trailing edge along the upper surface to the
leading edge, then along the lower surface back to the trailing edge.

## Thickness distribution

Both series share the thickness distribution of Abbott and von Doenhoff
(1959, section 6.4). With the thickness ratio $t$ [-] and the station $x$
[-], the half thickness $y_t$ [-] is

```{math}
y_t = 5 t (0.2969 \sqrt{x} - 0.1260 x - 0.3516 x^2 + 0.2843 x^3 + a_4 x^4).
```

The open trailing edge uses $a_4 = -0.1015$ and keeps a finite trailing
edge thickness. The closed trailing edge uses $a_4 = -0.1036$, which closes
the trailing edge to zero thickness (Ladson, Brooks, Hill and Sproles,
NASA TM-4741, 1996).

## 4-digit mean line

The 4-digit mean line follows Abbott and von Doenhoff (1959, section 6.4).
With the maximum camber $m$ [-] at the position $p$ [-], the mean line
$y_c$ [-] and its slope are, for $x < p$,

```{math}
y_c = \frac{m}{p^2}(2 p x - x^2), \qquad
\frac{dy_c}{dx} = \frac{2m}{p^2}(p - x),
```

and for $x \ge p$,

```{math}
y_c = \frac{m}{(1-p)^2}((1 - 2p) + 2 p x - x^2), \qquad
\frac{dy_c}{dx} = \frac{2m}{(1-p)^2}(p - x).
```

The designation "2412" gives $m = 0.02$, $p = 0.4$ and $t = 0.12$. The
designation "0012" gives $m = 0$ and $p = 0$ (symmetric section).

## 5-digit mean line

The standard (non-reflex) 5-digit mean lines follow Abbott and von Doenhoff
(1959). Jacobs, Ward and Pinkerton (NACA Report 460, 1933) and Jacobs and
Pinkerton (NACA Report 537, 1935) define the series. The first digit sets
the design lift coefficient ($0.15$ times the digit), the second and third
digits set the camber position, and the last two digits set $t$. With the
parameters $m$ [-] and $k_1$ [-], the mean line is, for $x \le m$,

```{math}
y_c = \frac{k_1}{6}(x^3 - 3 m x^2 + m^2 (3 - m) x), \qquad
\frac{dy_c}{dx} = \frac{k_1}{6}(3 x^2 - 6 m x + m^2 (3 - m)),
```

and for $x > m$,

```{math}
y_c = \frac{k_1 m^3}{6}(1 - x), \qquad
\frac{dy_c}{dx} = -\frac{k_1 m^3}{6}.
```

The parameters for design lift 0.3 (first digit 2) are:

| Mean line | m | k1 |
| --- | --- | --- |
| 210 | 0.0580 | 361.40 |
| 220 | 0.1260 | 51.64 |
| 230 | 0.2025 | 15.957 |
| 240 | 0.2900 | 6.643 |
| 250 | 0.3910 | 3.230 |

Other design lifts scale $k_1$ linearly with the first digit: a mean line
with first digit $d$ uses $k_1(d) = k_1(2) \times d / 2$. Reflex mean lines
(third digit 1) are not built: no published table of $k_1$ and $k_2/k_1$ is
loaded, so they raise an error.

## Coordinates

The thickness acts perpendicular to the mean line. With
$\theta = \atan(dy_c/dx)$, the upper surface is

```{math}
x_u = x - y_t \sin\theta, \qquad y_u = y_c + y_t \cos\theta,
```

and the lower surface uses the opposite signs. Cosine spacing sets
$x = (1 - \cos\beta)/2$ with uniform $\beta$ in $[0, \pi]$; uniform spacing
is also available. An odd point count gives $(n+1)/2$ stations per surface,
with the leading edge point stored once. A closed trailing edge repeats the
same point at the start and the end. The points are not re-normalized: the
perpendicular thickness moves the true leading edge of a cambered section
slightly off (0, 0), which the docstring states.

## Implementation

- `ventorum.airfoil.naca`: `naca4_thickness`, `naca4_camber`,
  `naca5_camber`, `naca4`, `naca5`, `naca`.

## References

- I. H. Abbott and A. E. von Doenhoff, *Theory of Wing Sections*, McGraw-Hill, 1959, section 6.4.
- C. L. Ladson, C. W. Brooks, A. S. Hill and D. W. Sproles, "Computer program to obtain ordinates for NACA airfoils", NASA TM-4741, 1996.
- E. N. Jacobs, K. E. Ward and R. M. Pinkerton, "The characteristics of 78 related airfoil sections from tests in the variable-density wind tunnel", NACA Report 460, 1933.
- E. N. Jacobs and R. M. Pinkerton, "Tests in the variable-density wind tunnel of related airfoils having the maximum camber unusually far forward", NACA Report 537, 1935.
