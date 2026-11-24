# Unsteady vortex-ring lattice and dynamic derivatives

## Model

The harmonic vortex-ring lattice solver models small-amplitude sinusoidal pitch and heave motions of a lifting surface or aircraft about a steady mean flight state, in free air or in flat-ground effect. Each panel carries a quadrilateral vortex ring. Behind the trailing edge, a flat harmonic wake of discrete vortex-ring rows marches downstream at the free-stream speed and is closed to infinity by an exact 3D semi-infinite harmonic far-wake tail. Because the frequency-dependent wake enters only the trailing-edge columns of the influence matrix, each reduced frequency is solved by a rank-$n_s$ Woodbury update to a single real LU factorisation of the bound influence matrix. Linearised Kutta-Joukowski loads, rotated mean loads, apparent-mass loads and a trailing-edge boundary correction yield the complex frequency-domain force and moment responses and the equivalent oscillatory longitudinal derivatives.

## Equations

### Vortex-ring lattice and ground images

A vortex lattice (`collocation="vlm"`) with $n_s$ spanwise strips and $n_c$ chordwise panels per strip maps to $N = n_s n_c$ quadrilateral vortex rings (`ventorum.unsteady.rings.build_ring_lattice`; Katz and Plotkin 2001, chapters 12 and 13). For each ring $p$ on strip $s$ and chordwise index $j \in \{0, \dots, n_c - 1\}$:

- The leading bound segment connects the panel quarter-chord endpoints $(\mathbf a_p, \mathbf b_p)$.
- For interior panels ($j < n_c - 1$), the rear corners coincide with the leading corners $(\mathbf a_{p+1}, \mathbf b_{p+1})$ of the next panel aft on the same strip.
- For the trailing-edge panel ($j = n_c - 1$), the vector from the bound segment to the physical trailing edge spans $0.75\,\Delta x_{\text{TE},s}$ along each strip edge. By default, the rear corners close at $0.25\,\Delta x_{\text{TE},s}$ aft of the physical trailing edge along each strip edge (Katz and Plotkin 2001, chapter 13), so adjacent strips share their common trailing-edge corner with zero gap.

Because adjacent chordwise rings share interior bound segments with opposite orientation, the equivalent horseshoe circulation on each strip relates to the ring circulation $\Gamma_p$ [m$^2$/s] by

```{math}
\gamma_{s,0} = \Gamma_{s,0}, \qquad \gamma_{s,j} = \Gamma_{s,j} - \Gamma_{s,j-1} \quad (j \ge 1).
```

Over flat ground, every bound ring and every wake strip receives a mirror image reflected in the ground plane with circulation sign $-1$, enforcing zero normal velocity on the ground plane (method of images, Katz and Plotkin 2001; see [Ground effect](ground_effect.md)).

### Harmonic wake and 3D far-wake tail

All strips share one discrete wake row length $\Delta x_w$ [m] and one convective time step $\Delta t$ [s], defined as the strip-width-weighted mean of the trailing-edge panel lengths:

```{math}
\Delta x_w = \frac{\sum_{s=1}^{n_s} w_s\,\Delta x_{\text{TE},s}}{\sum_{s=1}^{n_s} w_s}, \qquad \Delta t = \frac{\Delta x_w}{V_\infty}.
```

For a uniform-chord wing, $\Delta x_w = \Delta x_{\text{TE},s}$ identically. For a tapered wing, this avoids wake-row explosion at a narrow tip while recording the per-strip ratio $\Delta x_{\text{TE},s} / \Delta x_w$ for trust diagnostics.

Under harmonic motion at angular frequency $\omega$ [rad/s] and reduced frequency $k = \omega c_\text{ref} / (2 V_\infty)$ [-], the discrete wake consists of $N_{wx} = \max(1, \operatorname{round}(L_w / \Delta x_w))$ rows marching along the unit wake direction $\hat{\mathbf d}_s$. By Kelvin's circulation theorem and convection at $V_\infty$, wake row $m \in \{1, \dots, N_{wx}\}$ behind strip $s$ carries complex circulation (Hall 1994; Katz and Plotkin 2001, chapter 13)

```{math}
\Gamma_{w,s,m} = \Gamma_{\text{TE},s}\,e^{-\mathrm i \omega m \Delta t}.
```

Beyond row $N_{wx}$ (distance $L_w = N_{wx}\Delta x_w$), the discarded semi-infinite harmonic wake sheet on $\xi \in [0, \infty)$ carries doublet circulation $\Gamma_{\text{tail},s}(\xi) = \Gamma_{0,s}\,e^{-\mathrm i \kappa \xi}$ with spatial wavenumber $\kappa = \omega / V_\infty = 2 k / c_\text{ref}$ [1/m] and midpoint-matched phase anchor $\Gamma_{0,s} = \Gamma_{\text{TE},s}\,e^{-\mathrm i \omega (N_{wx} + 1/2)\Delta t}$ (`ventorum.unsteady.tail`). By Stokes' theorem, this semi-infinite sheet decomposes without far-field approximation into three parts per strip:

1. A closing spanwise vortex segment at $\xi = 0$ from $\mathbf p_{L,s}$ to $\mathbf p_{R,s}$.
2. Two semi-infinite oscillatory trailing vortex filaments along $+\hat{\mathbf d}_s$ from $\mathbf p_{R,s}$ ($+e^{-\mathrm i \kappa \xi}$) and $\mathbf p_{L,s}$ ($-e^{-\mathrm i \kappa \xi}$).
3. A continuous oscillatory shed vorticity sheet of density $\gamma_\text{shed}(\xi) = -\mathrm i \kappa\,e^{-\mathrm i \kappa \xi}$ integrated analytically across the span at fixed $\xi$.

When the tail start lies downstream of the evaluation point along $\hat{\mathbf d}_s$, all poles and branch points lie strictly in the left half-plane $\mathrm{Re}(\xi) < 0$. Rotating the integration contour to $\xi = -\mathrm i u$ ($u \in [0, \infty)$) converts the oscillatory factor $e^{-\mathrm i \kappa \xi}$ into the real decaying exponential $e^{-\kappa u} \in (0, 1]$, which is integrated by 32-point Gauss-Legendre quadrature under $u = L \tan\theta$. At $k = 0$ ($\kappa = 0$), the tail reduces identically to the semi-infinite steady wake closure.

### Rank-$n_s$ Woodbury frequency solve

Let $\mathbf A_{bb} \in \mathbb R^{N \times N}$ [1/m] be the real bound-ring normal-velocity influence matrix (including ground images), and let $\mathbf A_{bw} \in \mathbb R^{N \times N_{wx} \times n_s}$ [1/m] be the real influence tensor of the discrete wake rows. Both depend only on the geometry and ground height and are assembled once. For each reduced frequency $k$, the complex wake normal-velocity matrix $\mathbf W(k) \in \mathbb C^{N \times n_s}$ is

```{math}
\mathbf W(k) = \sum_{m=1}^{N_{wx}} e^{-\mathrm i \omega m \Delta t}\,\mathbf A_{bw,:,m,:} + e^{-\mathrm i \omega (N_{wx} + 1/2)\Delta t}\,\mathbf W_\text{tail}(k).
```

Because wake shedding couples only to the $n_s$ trailing-edge rings (selection matrix $\mathbf E_\text{te} \in \mathbb R^{N \times n_s}$), the harmonic impermeability system has a rank-$n_s$ complex update (Parenteau 2021, chapter 4):

```{math}
\left(\mathbf A_{bb} + \mathbf W(k)\,\mathbf E_\text{te}^T\right)\hat{\boldsymbol\Gamma}(k) = \hat{\mathbf b}(k).
```

$\mathbf A_{bb}$ is LU-factorised once in real float64 arithmetic. For each $k$, the solver computes $\mathbf Y = \mathbf A_{bb}^{-1}\hat{\mathbf b}(k)$ and $\mathbf Z = \mathbf A_{bb}^{-1}\mathbf W(k)$ by real triangular back-substitution on the real and imaginary parts, solves the small $n_s \times n_s$ capacitance system

```{math}
\left(\mathbf I_{n_s} + \mathbf E_\text{te}^T \mathbf Z\right)\mathbf c = \mathbf E_\text{te}^T \mathbf Y,
```

and recovers $\hat{\boldsymbol\Gamma}(k) = \mathbf Y - \mathbf Z\,\mathbf c$. The result agrees with a direct $N \times N$ complex LU solve to round-off while avoiding $O(N^3)$ complex factorisation at every frequency.

### Loaded-mean-state terms and travelling-wave wake displacement

In free air with zero mean lift ($\boldsymbol\Gamma_0 = \mathbf 0$), only the kinematic boundary velocity drives the perturbation circulation. About a lifting mean state ($\boldsymbol\Gamma_0 \neq \mathbf 0$), small heave $\delta z$ or pitch $\delta\theta$ moves the wing relative to the ground plane (in ground effect) and relative to the steady wake.

In the steady limit ($k = 0$), linearising $\mathbf A_\text{steady}(h, \theta)\,\boldsymbol\Gamma = \mathbf b$ about the mean state adds the rigid geometric right-hand-side forcing terms (Parenteau 2021, chapter 4)

```{math}
\mathbf b_{h,\text{geo}}^{(0)} = -c_\text{ref}\,\frac{\partial \mathbf A_\text{steady}}{\partial h}\,\boldsymbol\Gamma_0, \qquad
\mathbf b_{\theta,\text{geo}}^{(0)} = -\frac{\partial \mathbf A_\text{steady}}{\partial \theta}\,\boldsymbol\Gamma_0,
```

together with the matching derivatives of the steady induced velocity at the load evaluation points. In pitch, the mean aerodynamic force $\mathbf F_0$ and moment $\mathbf M_0$ about the reference point also rotate with the body ($\hat{\mathbf e}_y \times \mathbf F_0$ and $\hat{\mathbf e}_y \times \mathbf M_0$). Together, these $k = 0$ terms guarantee that the real parts at $k \to 0$ reproduce the steady height and incidence derivatives of `ventorum.height_derivatives` and `ventorum.analyze`.

At finite reduced frequency ($k > 0$), a prescribed wake does not move rigidly with the wing. Each wake corner remains where the trailing-edge closure was when it was shed and convects downstream with the free stream (Katz and Plotkin 2001, chapter 13). Linearising the prescribed-wake shedding geometry of the time-domain solver (`ventorum.unsteady.timedomain`) about the steady mean wake gives each steady wake station $m$ (at downstream distance $\xi_m = m \Delta x_w$) a travelling-wave displacement with the convective phase lag $e^{-\mathrm i \omega m \Delta t}$:

```{math}
\Delta\mathbf X_{e,s,m}^{\text{trav}}(k) = \mathbf u_{e,s}\left(e^{-\mathrm i \omega m \Delta t} - 1\right),
```

relative to the rigid $k = 0$ wake displacement $\mathbf u_{e,s}$, where $\mathbf u_{e,s} = c_\text{ref}\mathbf e_h$ for unit nondimensional heave and $\mathbf u_{e,s} = \hat{\mathbf e}_y \times (\mathbf p_{\text{TE},e,s} - \mathbf x_\text{ref})$ for unit pitch about $\mathbf x_\text{ref}$ ($e \in \{L, R\}$). In flat-ground effect, the mirror image of the displaced steady wake receives the reflected displacement $R_0(\Delta\mathbf X_{e,s,m}^{\text{trav}})$. Because every discrete steady wake row carries the same circulation $\Gamma_{0,\text{TE},s}$, all interior spanwise segments cancel identically, leaving the displaced trailing legs of the $N_{wx}$ discrete wake rows plus the two semi-infinite displaced far-wake trailing filaments on $\xi \in [0, \infty)$:

```{math}
\Delta\mathbf X_{e,s}^{\text{far}}(\xi; k) = \mathbf u_{e,s}\left(e^{-\mathrm i \kappa (L_w + \xi)} - 1\right).
```

The solver precomputes the per-station discrete wake normal-velocity and load-functional derivatives once by central finite differences of the ring kernels, contracts them per $k$ with $\phi_m = e^{-\mathrm i \omega m \Delta t} - 1$, and adds the semi-infinite far-wake contribution by contour-rotated quadrature along $\xi = (1 - \mathrm i) u$ (`ventorum.unsteady.tail`). This correction vanishes identically at $k = 0$ and at $\boldsymbol\Gamma_0 = \mathbf 0$, and makes the harmonic solver the exact small-amplitude periodic limit of the prescribed-wake time-domain solver at lifting mean states.

### Linearised loads and trailing-edge load correction

At each reduced frequency $k$, the linearised aerodynamic force and moment combine four physical contributions:

1. **Kutta-Joukowski loads** on all bound spanwise segments and chordwise legs: perturbation circulation times mean local velocity, plus mean circulation times perturbation velocity (motion velocity, geometric velocity derivative and perturbation induced velocity).
2. **Rotated mean loads** in pitch: rotation of the mean force and moment arm about the moment reference point.
3. **Apparent-mass loads** from the unsteady potential time derivative $\partial\phi / \partial t$ in the unsteady Bernoulli equation (Katz and Plotkin 2001, chapter 13):

```{math}
\mathbf F_{\text{am},p} = \rho\,(\mathrm i \omega\,\hat\Gamma_p)\,\mathbf A_p,
```

where $\mathbf A_p$ [m$^2$] is the physical surface area vector of bound ring $p$, spanning from its bound segment to the next bound segment on interior panels and to the physical trailing edge on the last panel ($0.75\,\Delta x_{\text{TE},s}$).

4. **Trailing-edge boundary load correction.** On the last bound panel, the trailing-edge ring closes at $0.25\,\Delta x_{\text{TE},s}$ aft of the physical trailing edge, while the first discrete wake row begins there with circulation $\Gamma_{w,s,1} = \Gamma_{\text{TE},s}\,e^{-\mathrm i \omega \Delta t}$. Across the junction, the discrete jump $\Gamma_{w,s,1} - \Gamma_{\text{TE},s}$ lies downstream of the physical trailing edge rather than distributed continuously up to the trailing edge. Without correction, the pitching-moment response converges only at first order in $1 / n_c$. Applying the bound-vorticity correction

```{math}
\Delta\gamma_{\text{TE},s} = f_{\text{TE},s}\left(\Gamma_{w,s,1} - \Gamma_{\text{TE},s}\right), \qquad
f_{\text{TE},s} = \frac{\Delta x_{\text{TE},s}}{2\left(\Delta x_{\text{TE},s} + \Delta x_w\right)},
```

at the physical trailing-edge segment $\mathbf l_{\text{TE},s}$ with the mean local velocity $\mathbf V_{\text{TE},0,s}$ gives $\Delta\mathbf F_{\text{TE},s} = \rho\,\Delta\gamma_{\text{TE},s}\,(\mathbf V_{\text{TE},0,s} \times \mathbf l_{\text{TE},s})$. Because $\Gamma_{w,s,1} = \Gamma_{\text{TE},s}$ at $k = 0$, the correction vanishes identically in steady flow and restores second-order chordwise convergence ($O(n_c^{-2})$) of unsteady pitching-moment derivatives.

### Equivalent oscillatory derivative definitions

Nondimensional time is $\tau = 2 V_\infty t / c_\text{ref}$ [-], reduced frequency is $k = \omega c_\text{ref} / (2 V_\infty)$ [-], and angular and incidence rates are normalised by $c_\text{ref} / (2 V_\infty)$ [s] as in [Body rates](rotation_rates.md). The two fundamental motions are:

- **Pitch** $\theta(t) = \mathrm{Re}(\hat\theta\,e^{\mathrm i \omega t})$ [rad] (positive nose up) about `ref_point` at constant height, giving complex response $H_\theta(k)$ [1/rad].
- **Heave** $z(t) = \mathrm{Re}(\hat z\,e^{\mathrm i \omega t})$ [m] (positive up along the lift direction in free air or the ground normal in ground effect) at constant attitude, giving complex response $H_z(k)$ [-] per unit $z / c_\text{ref}$.

For each coefficient $C \in \{C_L, C_D, C_Y, C_l, C_m, C_n\}$ (forces along mean wind axes, moments in body or stability axes), the equivalent oscillatory derivatives at reduced frequency $k$ (Rodden and Giesing 1970) are:

```{math}
C_\alpha(k) = \mathrm{Re}\,H_\theta(k) \quad [1/\text{rad}],
```

```{math}
C_q(k) = \frac{\mathrm{Im}\,H_{\text{pure}\,q}(k)}{k} \quad [-], \qquad
C_{\dot\alpha}(k) = \frac{\mathrm{Im}\,H_\theta(k)}{k} - C_q(k) \quad [-],
```

```{math}
C_h(k) = \mathrm{Re}\,H_z(k) \quad [-], \qquad
C_{\dot h}(k) = \frac{\mathrm{Im}\,H_z(k)}{2k} \quad [-], \qquad
C_{\dot h,\text{ground}}(k) = C_{\dot h}(k) + C_\alpha(k) \quad [-].
```

Here $C_h$ is per unit $h / c_\text{ref}$, $C_{\dot h}$ is per unit $(\mathrm dz/\mathrm dt) / V_\infty$, $C_q$ is per unit $q c_\text{ref} / (2 V_\infty)$, and $C_{\dot\alpha}$ is per unit $(\mathrm d\alpha/\mathrm dt) c_\text{ref} / (2 V_\infty)$.

**Pure pitch-rate motion.** Combining pitch $\theta$ with heave $z = V_\infty \theta / (\mathrm i \omega)$ (that is, $z / c_\text{ref} = \theta / (2 \mathrm i k)$) keeps the incidence $\alpha = \theta - (\mathrm i \omega / V_\infty) z = 0$ identically. For an unloaded mean state (in free air or in ground effect at zero lift), $H_{\text{pure}\,q}(k) = H_\theta(k) + H_z(k) / (2 \mathrm i k)$ gives the pure pitch-rate response directly. In loaded ground effect ($C_h \neq 0$), a heave amplitude proportional to $1 / (2k)$ introduces a static height term $-C_h / (2 k^2)$ into $\mathrm{Im}(H_z / (2\mathrm i k)) / k$; therefore, for a loaded mean state, `ventorum.dynamic_derivatives` evaluates $H_{\text{pure}\,q}(k)$ directly from the pure pitch-rate boundary condition at constant height on the same factorised Woodbury system. This agrees with $H_\theta + H_z / (2\mathrm i k)$ to round-off at zero mean load and approaches the steady pitch-rate derivative smoothly as $k \to 0$.

**Limit of the loaded pure pitch-rate response.** The direct pure pitch-rate solve at a loaded mean state uses the kinematic rotation velocity at the collocation points only. It does not add the loaded-mean-state geometric terms of the pitch and heave responses: the movement of the bound rings relative to the steady wake and the travelling-wave displacement of the steady wake. The wake therefore stays straight in this response. This is the model of the steady pitch-rate derivative of `rate_derivatives` (rotation about `ref_point` with a straight wake), so the $k \to 0$ limits agree. The size of the omitted terms at a loaded mean state is not measured; $C_q(k)$ and $C_{\dot\alpha}(k)$ at a loaded mean state carry this model limit.

**Low-frequency ($k \to 0$) limits.** In three dimensions, all six derivative groups have finite $k \to 0$ limits. `ventorum.dynamic_derivatives` always reports the $k \to 0$ limits in `k0`:

- $C_\alpha(0)$, $C_q(0)$ and $C_h(0)$ come from the exact real $k = 0$ solves of the harmonic ring system (`k0_pitch`, `k0_pitch_rate`, `k0_heave`).
- $C_{\dot\alpha}(0)$ and $C_{\dot h}(0)$ are evaluated at the small-frequency probe $k_\text{probe} = 10^{-3}$ (`K_ZERO_PROBE`), where the 32-point far-wake tail quadrature stays well conditioned even at high aspect ratio.
- $C_{\dot h,\text{ground}}(0) = C_{\dot h}(0) + C_\alpha(0)$. In free air at zero mean load, $C_h = 0$ and $C_{\dot h} + C_\alpha = 0$ as $k \to 0$.

**Stability axes.** With `axes="stability"`, forces remain relative to the mean free stream and moments rotate by the mean angle of attack $\alpha$ about the body $y$ axis via `ventorum.solvers.rate_derivatives.rate_axes_transform`. Because the rotation is about $y$, the pitch rate $q$ and pitching moment $C_m$ are invariant, while $(C_l, C_n)$ rotate by $\alpha$.

### Harmonic control-surface mode

For each named chordwise control surface (`n_chord >= 2`), the solver adds a unit harmonic deflection mode about the built lattice deflection $\delta_0$ [rad]:

```{math}
\delta(t) = \delta_0 + \mathrm{Re}\left(\hat\delta\,e^{\mathrm i \omega t}\right), \qquad \hat\delta = 1\ \text{rad}.
```

The mode uses the same strip hinge points $\mathbf H_{L,s}, \mathbf H_{R,s}, \mathbf H_{f,s}$ [m], unit hinge axes $\hat{\mathbf a}_{L,s}, \hat{\mathbf a}_{R,s}, \hat{\mathbf a}_{f,s}$ [-], sealed join edge weights $(c_{L,s}, c_{R,s})$, bent hinge-crossing side polylines and `ControlSurface.symmetric` sign rule as the moving-flap geometry of `ventorum.unsteady.timedomain` (`_build_control_rig`, `_deform_body_geometry`, `_compute_flap_velocities`). Nodes aft of the hinge line rotate about the strip hinge axis; hinge nodes on bent side polylines stay fixed.

**Boundary condition and geometric terms.** Linearising the impermeability condition about the steady mean state $(\delta_0, \boldsymbol\Gamma_0)$ gives the complex right-hand side on the same Woodbury system $\left(\mathbf A_{bb} + \mathbf W(k)\,\mathbf E_\text{te}^T\right)\hat{\boldsymbol\Gamma}_\delta = \hat{\mathbf b}_\delta(k)$:

```{math}
\hat{\mathbf b}_\delta(k)
= -\mathbf V_0 \cdot \frac{\partial \mathbf n_{bc}}{\partial \delta}
+ \mathrm i \omega \left(\mathbf v_{\text{fl},cp} \cdot \mathbf n_{bc,0}\right)
- \frac{\partial \mathbf A_\text{steady}}{\partial \delta}\,\boldsymbol\Gamma_0
- \Delta\mathbf v_{n,\text{wake}}^{\text{trav}}(k),
```

where:

- On each panel $p$ aft of the hinge on strip $s$, the unit normal rotates at rate $\partial \mathbf n_{bc,p} / \partial \delta = (c_{f,s}\hat{\mathbf a}_{f,s}) \times \mathbf n_{bc,p,0}$ with $c_{f,s} = c_{L,s} + f_{cp,s}(c_{R,s} - c_{L,s})$.
- The unit hinge-rate velocity is $\mathbf v_{\text{fl},cp,p} = (c_{f,s}\hat{\mathbf a}_{f,s}) \times (\mathbf p_{cp,p} - \mathbf H_{f,s})$ [m/rad] at collocation points aft of the hinge, giving surface velocity $\mathrm i \omega\,\mathbf v_{\text{fl},cp,p}$ [m/(s rad)].
- For a loaded mean state ($\boldsymbol\Gamma_0 \neq \mathbf 0$), the rigid $k = 0$ geometric influence term $-(\partial \mathbf A_\text{steady} / \partial \delta)\,\boldsymbol\Gamma_0$ captures the movement of the flap bound rings, bent hinge side polylines, flap collocation points and normals, trailing-edge steady wake start corners, and their flat-ground images. It is evaluated by central finite differences of `_deform_body_geometry` with default step $\varepsilon_\delta = 10^{-5}$ rad (`fd_step_delta`, clamped near the ground so $\varepsilon_\delta L_\text{max} \le 0.01 h_\text{min}$).
- The trailing-edge closure corners move by $\mathbf u_{\text{flap},L,s} = (c_{L,s}\hat{\mathbf a}_{L,s}) \times (\mathbf p_{\text{TE},L,s} - \mathbf H_{L,s})$ and $\mathbf u_{\text{flap},R,s} = (c_{R,s}\hat{\mathbf a}_{R,s}) \times (\mathbf p_{\text{TE},R,s} - \mathbf H_{R,s})$ [m/rad]. At finite frequency ($k > 0$), the shed steady wake acquires the same travelling-wave displacement $\mathbf u_{\text{flap},e,s}(e^{-\mathrm i \omega m \Delta t} - 1)$ on the discrete wake rows and $\mathbf u_{\text{flap},e,s}(e^{-\mathrm i \kappa (L_w + \xi)} - 1)$ on the semi-infinite far wake (with reflected displacements in ground effect).

**Linearised loads.** The control-mode force and moment combine:

1. Perturbation circulation $\hat{\boldsymbol\Gamma}_\delta$ times mean local velocity on the mean bound segments and chordwise legs.
2. Mean circulation $\boldsymbol\Gamma_0$ times the dynamic perturbation velocity (hinge-rate velocity $-\mathrm i \omega\,\mathbf v_{\text{fl,load}}$, perturbation bound and wake induced velocities, and the travelling-wave steady wake load delta).
3. Mean circulation $\boldsymbol\Gamma_0$ times the deformed flap geometry and steady induced velocity gradient, evaluated by elementwise central finite differences of the mean Kutta-Joukowski segment forces and moments at $\delta_0 \pm \varepsilon_\delta$ (capturing rotated flap legs, rotated leg-force normals, moved load points, moved moment arms, and ground images).
4. Apparent-mass loads $\rho\,(\mathrm i \omega \hat\Gamma_{\delta,p})\,\mathbf A_{p,0}$ over the physical ring areas (including bent hinge rings) and the trailing-edge boundary correction at the mean trailing edge. Because $\dot{\boldsymbol\Gamma}_0 = \mathbf 0$ and $\Delta\gamma_{\text{TE},0} = 0$, the first-order variations of $\mathbf A_p(\delta)$ and $\mathbf l_\text{TE}(\delta)$ do not enter these two terms.

**Equivalent oscillatory control derivatives and $k \to 0$ limit.** For each coefficient $C \in \{C_L, C_D, C_Y, C_l, C_m, C_n\}$ and control name `<name>`, let $H_\delta(k)$ [1/rad] be the complex response per unit deflection. The equivalent oscillatory control and control-rate derivatives (Rodden and Giesing 1970; Etkin and Reid 1996) are:

```{math}
C_{\delta}(k) = \mathrm{Re}\,H_\delta(k) \quad [1/\text{rad}], \qquad
C_{\dot\delta}(k) = \frac{\mathrm{Im}\,H_\delta(k)}{k} \quad [1/\text{rad}],
```

where the nondimensional deflection rate is $\dot\delta\,c_\text{ref} / (2 V_\infty)$ [rad]. At $k = 0$, $C_\delta(0)$ comes from the exact real steady solve $\mathbf A_\text{steady}\boldsymbol\Gamma_{\delta,0} = -\mathbf V_0 \cdot (\partial\mathbf n_{bc}/\partial\delta) - (\partial\mathbf A_\text{steady}/\partial\delta)\boldsymbol\Gamma_0$ (`k0_controls[name]`), which matches `ventorum.control_derivatives` on the same lattice. The $k \to 0$ control-rate limit $C_{\dot\delta}(0)$ is evaluated at the small-frequency probe $k_\text{probe} = 10^{-3}$ (`K_ZERO_PROBE`), matching $C_{\dot\alpha}(0)$ and $C_{\dot h}(0)$.

**Verification summary.** On a rectangular $AR = 6$ wing with a full-span 25 % chord flap (`hinge_x_c = 0.75`):

- At $k \to 0$ with `te_extension_fraction = 0.0` (`CORE_RADIUS_FRACTION = 1e-4`), `k0_controls["flap"]` agrees with `ventorum.control_derivatives` (`step_deg = 0.01`) to within $10^{-7}$ relative on $C_L, C_Y, C_l, C_m, C_n$ and near-field $C_D$ in free air and at $h/c = 0.5$ for $\alpha_0 = 0^\circ$ and $4^\circ$.
- Against the prescribed-wake time-domain solver (`_solve_time_rings`, $10^{-4}$ rad sinusoidal flap), the first harmonics of $C_L$ and $C_m$ agree within $0.1\%$ at $k \le 0.2$ and within $1\%$ at $k = 0.5$ in free air and at $h/c = 0.5$.
- On the same meshes ($n_c = 4, 8, 16$, $\Delta x_w = \Delta x_\text{TE}$) at $k = 0.5$, $\alpha_0 = 4^\circ$, the difference between the time-domain and harmonic flap first harmonics decreases at second order ($1.7 \le p \le 2.3$) for both $C_L$ and $C_m$, confirming the $O(\Delta t^2)$ accuracy of the second-order backward difference.
- On the documented moving-flap meshes at $\alpha_0 = 0^\circ$ (`n_chord = 4, 8, 16`), relative differences between time-domain and harmonic first harmonics are $1.35 \times 10^{-5}$ to $8.63 \times 10^{-6}$ ($C_L$) and $1.61 \times 10^{-5}$ to $8.70 \times 10^{-7}$ ($C_m$) at $k = 0.05$, and fall at second order from $6.43 \times 10^{-4}$ to $5.49 \times 10^{-5}$ ($C_L$) and $9.31 \times 10^{-4}$ to $6.43 \times 10^{-5}$ ($C_m$) at $k = 0.2$.

### Two-dimensional thin-airfoil limits

For a two-dimensional flat plate pitching about the quarter chord ($x_\text{ref} = c/4$, pivot parameter $a = -1/2$), Theodorsen (1935, NACA Report 496) gives the circulatory moment lever arm $a + 1/2 = 0$, so the pitching moment is independent of Theodorsen's circulation function $C(k)$:

```{math}
C_{m_q} = -\frac{\pi}{4}, \qquad
C_{m_{\dot\alpha}} = -\frac{\pi}{4}, \qquad
C_{m_q} + C_{m_{\dot\alpha}} = -\frac{\pi}{2},
```

and the quasi-steady lift-rate limit as $k \to 0$ is $C_{L_q} = C_{L_\alpha} = 2\pi$ (the three-quarter-chord incidence rule). For a finite-span wing of high aspect ratio (for example $AR = 40$), the 3D pitching-moment derivatives approach $-\pi / 4$ and $-\pi / 2$ while $C_{L_{\dot\alpha}}$ stays finite because of 3D tip relief (in 2D, $C(k)$ introduces a logarithmic divergence $\ln k$ in $C_{L_{\dot\alpha}}$ as $k \to 0$).

## Assumptions

- Incompressible, inviscid, attached potential flow (Mach $\le 0.3$).
- Thin lifting surfaces with small-amplitude harmonic pitch, heave and chordwise control-surface perturbations about a steady mean state.
- Flat, prescribed wake marching along the mean wake direction (no wake roll-up or free-wake deformation).
- Flat, rigid ground plane with no ground boundary layer when `condition.h` is set.
- Longitudinal rigid-body motions (pitch $\theta$ and heave $z$) plus named chordwise control deflections. Lateral rigid-body angular rates and sideslip rate are not yet modelled.

## Envelope

- **Solver scope.** Only the vortex-lattice solver (`solver_type="vlm"` or `"auto"`) is accepted. Lifting-line and Fourier solvers are refused with `InputError`.
- **Compressibility.** Mach numbers above `MACH_LIMIT_INCOMPRESSIBLE` ($0.3$), either stated in `condition.mach` or from $V_\infty / a$, are refused with `InputError`.
- **Ring-lattice geometry.** `ventorum.unsteady.rings.build_ring_lattice` accepts planar wings, cambered mean-line leg chains, deflected chordwise flaps (via bent side polylines) and joined surfaces (such as winglets). Lifting-line collocation (`collocation="llt"`) and section-mode (`n_chord = 1`) control modes are refused with `InputError`.
- **Ground clearance.** A configuration that touches or crosses the ground plane raises `GroundStrikeError`.
- **Trust notes.** The steady trust score and rating of the mean state (`base.totals.trust`) are preserved. Unsteady diagnostic rules add a warning when $n_c < 8$, and notes when $\Delta x_{\text{TE},s} / \Delta x_w \notin [0.8, 1.25]$, when $k > 1.0$, or when `wake_chords < 5.0` (see [Trust score and error bars](trust_score.md)).

## Implementation

- `ventorum.aero.vortex_ring`: numpy and Numba quadrilateral vortex-ring kernels and steady wake closure.
- `ventorum.unsteady.rings`: `RingLattice`, `RingGroundImage`, `build_ring_lattice`, `ground_image_rings`.
- `ventorum.unsteady.tail`: `harmonic_tail_velocity`, `harmonic_tail_normal_influence`, `displaced_steady_tail_velocity`, and `displaced_steady_tail_normal_influence` (closed-form 3D semi-infinite harmonic wake tail and travelling-wave displaced steady far wake with contour rotation).
- `ventorum.unsteady.harmonic`: `_HarmonicRingSystem`, `_HarmonicResponse`, `_solve_harmonic_rings`.
- `ventorum.unsteady.derivatives`: `dynamic_derivatives`, `DynamicDerivativesResult`.
- `ventorum.agent.tools.dynamic_derivatives` and `ventorum.agent.mcp_server`: agent tool and MCP server entry point.

## Verification

- [V16](../verification_report.md) in the verification report: steady $k \to 0$ limits (default and exact-equivalence settings), mirror-wing parity, Woodbury against direct complex LU, AR 40 mid-span heave lift against Theodorsen $C(k)$, chordwise convergence order with the trailing-edge correction, and 5-chord against 40-chord far-wake tail accuracy.
- `tests/test_unsteady_harmonic.py`: Woodbury against direct complex LU parity to round-off; explicit mirror-wing parity in ground effect with and without the 3D tail; exact-equivalence $k \to 0$ parity against `ventorum.analyze`, `height_derivatives` and `rate_derivatives`; loaded-mean-state finite-difference step independence; trailing-edge load correction (zero at $k = 0$ and second-order chordwise convergence of $\mathrm{Re}(C_{m_{\dot\alpha}})$); 5-chord tail accuracy against 40 chords; AR 40 mid-span strip lift against Theodorsen's $C(k)$.
- `tests/test_unsteady_flap_mode.py`: control-surface $k \to 0$ parity against `ventorum.control_derivatives`; Woodbury against direct LU and finite-difference step independence; ground mirror parity against the explicit two-wing model; antisymmetric aileron symmetry; periodic-limit agreement and second-order time-step convergence at $k = 0.5$ against `_solve_time_rings`; comparison with the documented moving-flap first harmonics.
- `tests/test_dynamic_derivatives.py`: default-settings $k \to 0$ parity against `ventorum.analyze`, `rate_derivatives`, `height_derivatives` and `control_derivatives`; loaded ground-effect smoothness of $C_q(k)$ and $C_{\dot\alpha}(k)$; free-air identities ($C_h = 0$ and $C_{\dot h} + C_\alpha = 0$); stability-axes rotation (including control modes); AR 40 rectangular wing against Theodorsen thin-airfoil limits ($-\pi/4$, $-\pi/2$, $2\pi$); strict JSON `to_dict()`; all input checks and all four trust rules.
- `tests/test_gpu_ring_kernels.py`: GPU against CPU parity in float64 and float32 for the ring kernels, far-wake tail, heave/pitch modes and control-surface modes.
- `tests/test_agent_dynamic_derivatives.py`: numeric parity between `ventorum.agent.tools.dynamic_derivatives`, MCP `tools/call` and `ventorum.dynamic_derivatives`.

## References

- B. Etkin and L. D. Reid, *Dynamics of Flight: Stability and Control*, 3rd ed., Wiley, 1996 (control and control-rate derivatives; cited, original text not read directly).
- J. Katz and A. Plotkin, *Low-Speed Aerodynamics*, 2nd ed., Cambridge University Press, 2001, chapters 12 and 13 (vortex-ring lattice, trailing-edge closure, unsteady Bernoulli loads and method of images; cited, original text not read directly).
- K. C. Hall, "Eigenanalysis of unsteady flows about airfoils, cascades, and wings", *AIAA Journal* 32(12), 1994, pp. 2426-2432 (harmonic wake phase relation; cited through Parenteau 2021, original paper not read directly).
- M. Parenteau, *A Non-Linear Frequency Domain Potential Flow Model for Compressible, Transonic and Viscous Aeroelastic Analyses*, PhD thesis, Polytechnique Montreal, 2021, chapter 4 (frequency-domain vortex lattice, rank-$n_s$ Woodbury update and loaded-mean-state terms; cited through module formulation, original thesis not read directly).
- W. P. Rodden and J. P. Giesing, "Application of oscillatory aerodynamic theory to estimation of dynamic stability derivatives", *Journal of Aircraft* 7(3), 1970, pp. 272-275 (equivalent derivatives at reduced frequency $k$; cited, original paper not read directly).
- T. Theodorsen, "General theory of aerodynamic instability and the mechanism of flutter", NACA Report 496, 1935 (2D thin-airfoil oscillatory lift, pitching moment and circulation function $C(k)$; cited, original report not read directly).
