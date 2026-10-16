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
a node at each active hinge, so both surfaces carry the kink of the flap.
Forward of the hinge the stations stay at their chord fractions. Aft of the
hinge the stations of each deflected surface are stretched linearly from the
hinge to its rotated trailing edge, so no station falls beyond the trailing
edge (the rotated trailing edge is forward of x = 1 for any deflection). The
trailing-edge nodes use the exact rotated trailing-edge point [chord units],
so the rotation about the hinge line holds to round-off.

On the compressed side of the hinge the moved surface can start forward of
the fixed hinge corner without a crossing of the fixed surface (for example
when the lower surface rises toward the trailing edge). The mesh drops the
moved points that lie forward of a point already kept, so each surface is
single valued in x. The shape changes only inside this small notch at the
hinge.

Each section lies in the plane of the local chord direction and the local
section normal (the normal of the vortex-lattice surface at the station), so
the deflection turns the flap about the local hinge line also on a surface
with dihedral. On the left half of a full mesh the section frames are the
mirror images of the right-side frames.

## Side closures

At the side edges of a deflected control the surface is not continuous. The
mesh carries two spanwise stations at the same y [m], one with the deflected
shape and one with the undeflected shape (or two different deflections). The
nodes that the two sections share (forward of the hinge, where the shapes are
equal) are common nodes. Aft of them, the two section outlines run from the
last common upper node around the trailing edge to the last common lower
node. These two chains bound the gap polygon in the plane of the station.

The closure is built in these steps:

1. Find all crossings of the two chains in the station plane. A proper
   crossing adds a new node on both chains. A node of one chain that lies on
   a segment of the other chain is put into that segment.
2. Walk the closed polygon (first chain, then the second chain backward). Each
   node that the walk meets twice closes a lobe. Each lobe is a simple
   polygon. Its orientation is the orientation of the walk, which gives the
   outward normal: a lobe where only the first section has material faces
   toward the second station, and the other lobes face the other way.
3. Triangulate each lobe by ear clipping, in its own orientation. Among the
   valid ears (convex, no other node inside or on the ear) the ear with the
   largest smallest angle is cut first. Each side panel is therefore a plane
   triangle with its collocation point inside.
4. A node that step 1 puts on a section outline also lies on the edge of the
   surface panel next to the closure. That panel is split into triangles
   (two-chain triangulation with the shorter diagonal), so the mesh has no
   hanging node and stays watertight.

The side panels have the kind code "side", strip index -1 and side 0. The
closure models no gap flow: air does not pass through the side gap of the
flap.

On the left half of a symmetric surface the deflection follows the
`symmetric` flag: the same deflection [rad] for `symmetric=True`, the opposite
deflection for `symmetric=False`. An antisymmetric control on a symmetric
surface needs the full mesh: `half=True` is refused for it, because the half
mesh assumes symmetry about y = 0. A non-symmetric mirror copy uses the left
side rule. Zero deflection gives the same mesh as before, bit for bit.

## Strip metadata

A mesh with controls has a different panel order than a plain mesh (the split
surface panels and the side panels). The solver must use the metadata fields
of `PanelMesh` for strip loads: `strip` (spanwise strip of each panel),
`chord_index` (chordwise index from the leading edge, the hinge station
included), `side` (+1 upper, -1 lower, 0 base, tip and side panels) and
`node_station` (spanwise station row of each node). The zero-width gap
between the two stations of a side edge is not a strip.

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

- zero deflection gives the same mesh (and metadata) bit for bit;
- 10 and -20 deg flaps over 40 to 80 percent semi-span stay watertight and
  outward, every panel is convex and planar, the side panels are triangles in
  the planes of the side edges, and the trailing-edge nodes inside the range
  are rotated about the hinge line while nodes outside are unchanged;
- the self-influence of every panel (doublet potential at its own
  collocation point) is +1/2 for flap and aileron meshes with library
  sections;
- the matrix condition number of the panel solve stays below 1e4;
- the flap delta CL and the aileron Cl from the circulation of the wake strips
  agree with the VLM (same planform, controls and n_chord 12 and 20) to 25
  percent, for the library sections 0012 and 2412;
- the pressure-integrated flap delta CL and aileron Cl agree with the VLM to
  25 percent for a section with a closed trailing edge. For library sections
  with a blunt trailing edge the pressure integration of the solver is not
  yet in this range (a strict expected-failure test records it);
- the full mesh of an antisymmetric control on a wing with dihedral, sweep and
  twist is the mirror image of itself outside the control span;
- the strip metadata gives one strip per spanwise panel with all chordwise
  indices on both sides.

## References

- H. Glauert, *The Elements of Aerofoil and Airscrew Theory*, Cambridge
  University Press, 1926, Chapter VII.
- H. Glauert, Theoretical relationships for an aerofoil with hinged flap,
  Aeronautical Research Committee, R and M 1095, 1927.
- I. H. Abbott and A. E. von Doenhoff, *Theory of Wing Sections*, Dover, 1959.
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge
  University Press, 2001.
- J. L. Hess and A. M. O. Smith, Calculation of non-lifting potential flow
  about arbitrary three-dimensional bodies, Journal of Ship Research, Vol. 8,
  No. 2, 1964, pp. 22-44.
- J. L. Hess and A. M. O. Smith, Calculation of potential flow about arbitrary
  bodies, Progress in Aeronautical Sciences, Vol. 8, 1967, pp. 1-138.
- J. O'Rourke, *Computational Geometry in C*, 2nd ed., Cambridge University
  Press, 1998, chapter 1 (polygon triangulation by ear clipping).
