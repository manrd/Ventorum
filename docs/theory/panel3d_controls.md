---
orphan: true
---

# Plain control surfaces in 3D panel meshes

This page describes deflected plain control surfaces in thick 3D panel meshes.

## Purpose

The vortex lattice method and the lifting lines model flaps and controls with
the `ControlSurface` input. The 3D panel method takes the same input: a plain
control surface with a hinge line, a chord fraction, a span range and a
deflection, meshed as a deflected thick surface.

## Geometry

Inside each control span range, the section shapes use
`ventorum.airfoil.flap.deflect_plain_flap` at the control chord fraction and
deflection. The deflection [rad] uses the same sign convention as the VLM
controls: positive moves the trailing edge down on a horizontal surface. The
hinge height uses the mid thickness fraction 0.5.

The chordwise resampling runs on the deflected shape. The target stations gain
a node at each active hinge, so both surfaces carry the kink of the flap. The
trailing-edge nodes use the exact rotated trailing-edge point [chord units],
so the rotation about the hinge line holds to round-off.

At the side edges of a deflected control the surface is not continuous. The
mesh carries two spanwise stations at the same y [m], one with the deflected
shape and one with the undeflected shape. The strip between the pair is a
flat side closure, so the mesh stays watertight. The closure models no gap
flow: air does not pass through the side gap of the flap.

On the left half of a symmetric surface the deflection follows the
`symmetric` flag: the same deflection [rad] for `symmetric=True`, the opposite
deflection for `symmetric=False`. A non-symmetric mirror copy uses the left
side rule. Zero deflection gives the same mesh as before, bit for bit.

## Solver effect

A positive (trailing edge down) flap deflection increases the lift coefficient
CL. An antisymmetric aileron pair gives a rolling moment with the sign of the
VLM for the same input: positive aileron deflection gives negative Cl (right
wing up) in the AVL body axes conventions.

## Assumptions and limits

The model uses these assumptions:

- Small deflection angles: the lattice controls assume |deflection| <= 30 deg;
  the section tool allows up to 60 deg.
- No gap flow at the control side edges; the flat closure seals the gap.
- No flap-edge vortex model other than the panel mesh itself.
- Attached inviscid flow only; no viscous separation effects.
- The hinge height stays at mid thickness; no nose balance.

## Verification

`tests/test_panel3d_controls.py` checks:

- zero deflection gives the same mesh bit for bit;
- 10 and -20 deg flaps over 40 to 80 percent semi-span stay watertight and
  outward, with trailing-edge nodes inside the range rotated about the hinge
  line and nodes outside unchanged;
- a panel solve (CPU) gives a CL increase for positive deflection;
- an antisymmetric aileron pair gives a rolling moment with the VLM sign.

## References

- H. Glauert, *The Elements of Aerofoil and Airscrew Theory*, Cambridge
  University Press, 1926, Chapter VII.
- H. Glauert, Theoretical relationships for an aerofoil with hinged flap,
  Aeronautical Research Committee, R and M 1095, 1927.
- I. H. Abbott and A. E. von Doenhoff, *Theory of Wing Sections*, Dover, 1959.
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge
  University Press, 2001.
- J. L. Hess and A. M. O. Smith, Calculation of potential flow about arbitrary
  bodies, Journal of Ship Research, Vol. 8, 1964.
