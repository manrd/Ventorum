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

## Side edges: sealed transition (default)

By default (`control_edge="transition"` of `mesh_lifting_surface`) the
deflection of a control changes linearly over a spanwise transition centred
on each side edge that borders an undeflected part of the span. The
transition is `control_transition` local chords long (`CONTROL_TRANSITION`,
0.2, a method choice). It is shortened where the control itself, the root,
the tip or the transition of the next edge leave less room. Each end of the
transition is a mesh station; the stations inside carry the deflection times
the ramp factor (0 at the outer end, 1 at the inner end). The surface stays
continuous: there is no side face and no gap, as for the camber step of a
vortex lattice at a control edge. An edge between two deflected controls
keeps the gap closure of the next section.

The reason is the flow through the side gap. Near the trailing edge of a gap
edge the two section outlines cross: a flap down puts the flap upper surface
below the lower surface of the fixed part. The region between the two
outlines in the station plane is then a window of fluid that joins the flow
under the fixed part to the flow over the flap. The potential difference
across the window is of the order of the circulation, so the potential flow
through the window is singular at its sharp edges, and the circulation of
the strips next to the edge falls as they become narrower. CL falls and the
Trefftz induced drag rises with spanwise refinement. The two trailing-edge
nodes of a gap edge are also not joined by a wake, so the two wake strips
end in two trailing vortices of nearly the full circulation instead of one
vortex of the circulation difference. Maskew (1987, section 4.4) prescribes
vortex separations at the flap edge. Two such models were tried: a wake
strip between the two trailing-edge nodes bounds the induced drag but does
not stop the loss of lift, and a wake panel over the window made the
solution unstable. The sealed transition converges with spanwise refinement,
and its converged result changes very little with the transition width; the
flap increment of CL is close to the increment of the vortex lattice (the
thick sections make the flap a little more effective). See
`tests/test_panel3d_controls.py`.

## Side edges: gap closure (option)

With `control_edge="gap"` (the former mesh), and always at an edge between
two deflected controls, the surface is not continuous at the side edge. The
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
between the two stations of a gap edge is not a strip. A transition edge adds
two stations, so the mesh can have more strips than the lattice; the solver
then assigns the panels to the lattice strips by their centroids.

## Solver effect

A positive (trailing edge down) flap deflection increases the lift coefficient
CL. An antisymmetric aileron pair gives a rolling moment with the sign of the
VLM for the same input: positive aileron deflection gives negative Cl (right
wing up) in the AVL body axes conventions.

## Assumptions and limits

The model uses these assumptions:

- Small deflection angles: the lattice controls assume |deflection| <= 30 deg;
  the section tool allows up to 60 deg.
- No gap flow at the control side edges: the default transition seals the
  edge with a continuous surface; the gap option closes the gap with flat
  side panels.
- No flap-edge vortex model other than the panel mesh itself.
- Attached inviscid flow only; no viscous separation effects.
- The hinge height stays at mid thickness; no nose balance.

## Verification

`tests/test_panel3d_controls.py` checks:

- zero deflection gives the same mesh (and metadata) bit for bit;
- with the default transition edges, a station lies at each end of each
  transition, there are no side panels, and every trailing-edge node is the
  trailing edge rotated about the hinge by the deflection times the ramp
  factor;
- a flapped wing of aspect ratio 8 converges with spanwise refinement: CL and
  the Trefftz CDi change by less than 2 percent from 48 to 96 strips, in the
  same direction as from 24 to 48 strips, and the flap increment of CL agrees
  with the VLM to 10 percent (slow test); a coarse version checks 24 against
  48 strips;
- with the gap option, 10 and -20 deg flaps over 40 to 80 percent semi-span
  stay watertight and outward, every panel is convex and planar, the side
  panels are triangles in the planes of the side edges, and the trailing-edge
  nodes inside the range are rotated about the hinge line while nodes outside
  are unchanged;
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
- B. Maskew, Program VSAERO Theory Document, NASA CR-4023, 1987, section 4.4
  (part-span flap with prescribed vortex separations at the flap edge).
