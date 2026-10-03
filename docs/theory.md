# Theoretical Background: SAF Skyrmion Simulator

## 1. Magnetic Skyrmions in Synthetic Antiferromagnets

Magnetic skyrmions are topologically non-trivial chiral spin textures with a
topological charge $S = \pm 1$. In a skyrmion, the magnetization rotates
smoothly from one out-of-plane direction at the core to the opposite direction
in the surrounding domain.

A synthetic antiferromagnet (SAF) consists of two ferromagnetic layers (Co)
separated by a non-magnetic spacer (Ru), coupled antiferromagnetically via the
Ruderman-Kittel-Kasuya-Yosida (RKKY) interaction. In a SAF, skyrmions in the
two layers have opposite core polarities and opposite topological charges,
leading to a cancellation of the Magnus force and a vanishing skyrmion Hall
effect [1].

The SAF stack modeled here is Pt/Co/Ru/Pt/Co/Ru, following the experimental
system in [1].

## 2. Hamiltonian

The total magnetic energy of the SAF system is the sum of contributions from
both layers ($\ell = \text{top}, \text{bot}$) plus the interlayer coupling:

$$
\mathcal{H} = \sum_{\ell} \left(
\mathcal{H}_{\text{ex}}^{(\ell)}
+ \mathcal{H}_{\text{DMI}}^{(\ell)}
+ \mathcal{H}_{\text{anis}}^{(\ell)}
+ \mathcal{H}_{\text{demag}}^{(\ell)}
+ \mathcal{H}_{\text{Z}}^{(\ell)}
\right)
+ \mathcal{H}_{\text{RKKY}}
$$

The effective field entering the LLGS equation is obtained from the variational
derivative of the energy with respect to the magnetization:

$$
\mathbf{H}_{\text{eff}} = -\frac{1}{M_s} \frac{\delta \mathcal{H}}{\delta \mathbf{m}}
$$

### 2.1. Exchange Energy

The Heisenberg exchange favors parallel alignment of neighboring spins:

$$
\mathcal{H}_{\text{ex}} = A_{\text{ex}} \int
\left[
(\nabla m_x)^2 + (\nabla m_y)^2 + (\nabla m_z)^2
\right] dV
$$

On the discrete 2D lattice with lattice constant $a$ and film thickness
$t_{\text{Co}}$:

$$
\mathcal{H}_{\text{ex}} = -2A_{\text{ex}} \, t_{\text{Co}}
\sum_{\langle i,j \rangle} \mathbf{m}_i \cdot \mathbf{m}_j
$$

where the sum runs over nearest-neighbor pairs, each counted once
(constant terms dropped).

### 2.2. Dzyaloshinskii-Moriya Interaction Energy

The interfacial (Neel-type) DMI at the Pt/Co interface introduces chirality:

$$
\mathcal{H}_{\text{DMI}} = D \, t_{\text{Co}} \int
\left[
m_z \, (\nabla \cdot \mathbf{m})
- (\mathbf{m} \cdot \nabla) m_z
\right] dS
$$

On the discrete lattice this becomes a sum over nearest-neighbor bonds with
bond-direction-dependent cross products:

$$
\mathcal{H}_{\text{DMI}} = -D \, t_{\text{Co}} \, a
\sum_{\langle i,j \rangle}
(\hat{r}_{ij} \times \hat{z}) \cdot (\mathbf{m}_i \times \mathbf{m}_j)
$$

where $\hat{r}_{ij}$ is the unit vector from site $i$ to site $j$ and
each bond is counted once.

### 2.3. Anisotropy Energy

Uniaxial perpendicular magnetic anisotropy (PMA) along the film normal:

$$
\mathcal{H}_{\text{anis}} = -K \, t_{\text{Co}} \int m_z^2 \, dS
$$

On the lattice:

$$
\mathcal{H}_{\text{anis}} = -K \, t_{\text{Co}} \, a^2 \sum_i m_{z,i}^2
$$

### 2.4. Demagnetization Energy

In the thin-film limit ($t_{\text{Co}} \ll$ lateral dimensions), the
magnetostatic energy reduces to a local shape anisotropy:

$$
\mathcal{H}_{\text{demag}} = \frac{\mu_0 M_s^2}{2} \, t_{\text{Co}}
\int m_z^2 \, dS
$$

This opposes out-of-plane magnetization. Combined with the PMA, it yields the
effective anisotropy $K_{\text{eff}} = K - \mu_0 M_s^2 / 2$.

### 2.5. Zeeman Energy

Interaction with an external field $\mathbf{B}_{\text{ext}}$:

$$
\mathcal{H}_{\text{Z}} = -M_s \, t_{\text{Co}} \int
\mathbf{m} \cdot \mathbf{B}_{\text{ext}} \, dS
$$

### 2.6. RKKY Interlayer Coupling Energy

The antiferromagnetic RKKY exchange between the two Co layers across the Ru
spacer:

$$
\mathcal{H}_{\text{RKKY}} = J_{\text{RKKY}} \, \int
\mathbf{m}_{\text{top}} \cdot \mathbf{m}_{\text{bot}} \, dS
$$

where $J_{\text{RKKY}} > 0$ for antiferromagnetic coupling. The coupling field
is related to the coupling constant by
$\mu_0 H_{\text{RKKY}} = J_{\text{RKKY}} / (M_s \, t_{\text{Co}})$.

### 2.7. Total Energy on the Discrete Lattice

Combining all contributions, the total energy per layer on the 2D lattice is:

$$
\mathcal{H}^{(\ell)} = t_{\text{Co}} \, a^2 \sum_i \Bigg[
-\frac{A_{\text{ex}}}{a^2} \sum_{\delta} \mathbf{m}_i \cdot \mathbf{m}_{i+\delta}
- \frac{D}{2a} \sum_{\delta} (\hat{\delta} \times \hat{z}) \cdot
(\mathbf{m}_i \times \mathbf{m}_{i+\delta})
- K_{\text{eff}} \, m_{z,i}^2
- M_s \, \mathbf{m}_i \cdot \mathbf{B}_{\text{ext}}
\Bigg]
$$

where $\delta$ runs over the four nearest neighbors $\{\pm\hat{x}, \pm\hat{y}\}$.
The site sum visits every bond twice, which halves the per-bond
coefficients above. The interlayer RKKY term adds
$J_{\text{RKKY}} \, a^2 \sum_i \mathbf{m}_{\text{top},i} \cdot \mathbf{m}_{\text{bot},i}$.

## 3. Landau-Lifshitz-Gilbert-Slonczewski Equation (LLGS)

The magnetization dynamics of each layer are governed by the
Landau-Lifshitz-Gilbert-Slonczewski (LLGS) equation. For the unit
magnetization vector $\mathbf{m} = \mathbf{M}/M_s$:

$$
\frac{d\mathbf{m}}{dt} = -\gamma \, \mathbf{m} \times \mathbf{H}_{\text{eff}}
+ \alpha \, \mathbf{m} \times \frac{d\mathbf{m}}{dt}
+ \gamma H_{\text{DL}} \, \mathbf{m} \times (\hat{\mathbf{p}} \times \mathbf{m})
+ \gamma H_{\text{FL}} \, \hat{\mathbf{p}} \times \mathbf{m}
$$

where:

- $\gamma$ is the gyromagnetic ratio (rad/s/T)
- $\alpha$ is the Gilbert damping constant
- $\mathbf{H}_{\text{eff}}$ is the effective magnetic field (Tesla)
- $H_{\text{DL}}$ is the damping-like (DL) spin-orbit torque field
- $H_{\text{FL}}$ is the field-like (FL) spin-orbit torque field
- $\hat{\mathbf{p}}$ is the spin polarization direction

These are the standard Slonczewski torques. Their magnitude is
proportional to $|\hat{\mathbf{p}} \times \mathbf{m}| = \sin\vartheta$,
the sine of the angle between $\mathbf{m}$ and $\hat{\mathbf{p}}$, so
they vanish when $\mathbf{m}$ is parallel to the spin polarization.

### 3.1. Explicit Form

The implicit form above is converted to an explicit equation by substituting
$d\mathbf{m}/dt$ into the damping term. Using $|\mathbf{m}| = 1$ and
$\mathbf{m} \cdot d\mathbf{m}/dt = 0$:

$$
\frac{d\mathbf{m}}{dt} = \frac{\gamma}{1 + \alpha^2} \Big[
-(\mathbf{m} \times \mathbf{H})
- \alpha \, \mathbf{m} \times (\mathbf{m} \times \mathbf{H})
+ (H_{\text{DL}} + \alpha H_{\text{FL}})(\mathbf{m} \times \mathbf{s})
+ (H_{\text{FL}} - \alpha H_{\text{DL}}) \, \mathbf{s}
\Big]
$$

where $\mathbf{s} = \hat{\mathbf{p}} \times \mathbf{m}$, and $H_{\text{DL}}$, $H_{\text{FL}}$ are the SOT effective fields in Tesla.

## 4. Effective Field Contributions

The total effective field for each layer is:

$$
\mathbf{H}_{\text{eff}} = \mathbf{H}_{\text{ex}} + \mathbf{H}_{\text{DMI}}
+ \mathbf{H}_{\text{anis}} + \mathbf{H}_{\text{Z}}
+ \mathbf{H}_{\text{RKKY}} + \mathbf{H}_{\text{demag}}
$$

All fields are computed in Tesla.

### 4.1. Exchange Field

The Heisenberg exchange interaction favors parallel alignment of neighboring
spins. On a 2D square lattice with lattice constant $a$, the discrete exchange
field is the lattice Laplacian:

$$
\mathbf{H}_{\text{ex}} = \frac{2 A_{\text{ex}}}{M_s \, a^2}
\sum_{\text{nn}} (\mathbf{m}_j - \mathbf{m}_i)
= \frac{2 A_{\text{ex}}}{M_s \, a^2}
(\mathbf{m}_{+x} + \mathbf{m}_{-x} + \mathbf{m}_{+y} + \mathbf{m}_{-y}
 - 4\mathbf{m})
$$

where $A_{\text{ex}}$ is the exchange stiffness constant.

### 4.2. Dzyaloshinskii-Moriya Interaction (DMI)

The interfacial (Neel-type) DMI arises at the Pt/Co interface and stabilizes
chiral spin textures. The continuum DMI energy density is:

$$
\varepsilon_{\text{DMI}} = D \left[ m_z \, \nabla \cdot \mathbf{m}
- (\mathbf{m} \cdot \nabla) m_z \right]
$$

The corresponding effective field, discretized with central finite differences
on the lattice:

$$
H_{\text{DMI}, x} = \frac{D}{M_s \, a} (m_{+x,z} - m_{-x,z})
$$

$$
H_{\text{DMI}, y} = \frac{D}{M_s \, a} (m_{+y,z} - m_{-y,z})
$$

$$
H_{\text{DMI}, z} = -\frac{D}{M_s \, a} (m_{+x,x} - m_{-x,x} + m_{+y,y} - m_{-y,y})
$$

where $D$ is the DMI constant. Positive $D$ favors left-handed Neel
skyrmions, consistent with the Pt/Co interface.

### 4.3. Perpendicular Magnetic Anisotropy

Uniaxial anisotropy along the film normal ($\hat{z}$):

$$
\mathbf{H}_{\text{anis}} = \frac{2 K}{M_s} \, m_z \, \hat{z}
$$

where $K$ is the anisotropy constant for the respective layer.

### 4.4. Thin-Film Demagnetization

In a thin film ($t_{\text{Co}} \ll$ lateral dimensions), the demagnetizing
field is well approximated by the local thin-film limit:

$$
\mathbf{H}_{\text{demag}} = -\mu_0 M_s \, m_z \, \hat{z}
$$

This opposes out-of-plane magnetization and effectively reduces the
anisotropy. The combined anisotropy + demagnetization field uses the effective
anisotropy:

$$
K_{\text{eff}} = K - \frac{\mu_0 M_s^2}{2}
$$

$$
\mathbf{H}_{\text{anis+demag}} = \frac{2 K_{\text{eff}}}{M_s} \, m_z \, \hat{z}
$$

The system in [1] is close to the spin reorientation transition
($K_{\text{eff}} \approx 0$), which favors skyrmion formation.

Beyond this local limit, the stray field can be computed from the full
demagnetizing tensor in Fourier space. The analytic thin-slab kernel
treats each layer as a continuous film; the Newell kernel treats each
cell as a finite rectangular prism, using the exact Newell-Williams-Dunlop
tensor near the source and the point-dipole tensor far from it. Both
couple the two layers, including cross terms in which the in-plane
magnetization of one layer produces an out-of-plane field in the
other. A finite sample is modelled by zero-padding the grid across its
free directions, so the Fourier convolution no longer wraps around.

### 4.5. Zeeman Field

A uniform external magnetic field $\mathbf{H}_{\text{ext}}$ (in Tesla),
identical at all lattice sites.

### 4.6. RKKY Interlayer Coupling

The antiferromagnetic RKKY interaction between the two Co layers promotes
antiparallel alignment:

$$
\mathbf{H}_{\text{RKKY, on top}} = -H_{\text{RKKY}} \, \mathbf{m}_{\text{bot}}
$$

$$
\mathbf{H}_{\text{RKKY, on bot}} = -H_{\text{RKKY}} \, \mathbf{m}_{\text{top}}
$$

where $H_{\text{RKKY}}$ is the RKKY coupling field in Tesla.

## 5. Spin-Orbit Torques

The spin-orbit torques (SOT) arise from the spin Hall effect at the Pt/Co
interface. An in-plane current density $J$ generates a spin accumulation with
polarization $\hat{\mathbf{p}}$ perpendicular to the current direction.

The damping-like (DL) and field-like (FL) SOT effective fields follow
the applied current pulse $J(t)$:

$$
H_{\text{DL}} = \chi_{\text{DL}} \cdot J(t)
$$

$$
H_{\text{FL}} = \chi_{\text{FL}} \cdot J(t)
$$

where $\chi_{\text{DL}}$ and $\chi_{\text{FL}}$ are the SOT coefficients
(in T A$^{-1}$ m$^2$) measured experimentally.

In the LLGS equation, DL-SOT enters as the damping-like torque
$\mathbf{m} \times (\hat{\mathbf{p}} \times \mathbf{m})$ and FL-SOT as the
field-like torque $\hat{\mathbf{p}} \times \mathbf{m}$.

## 6. Topological Charge

The topological charge (skyrmion number) is:

$$
Q = \frac{1}{4\pi} \int \mathbf{m} \cdot
\left( \frac{\partial \mathbf{m}}{\partial x} \times
\frac{\partial \mathbf{m}}{\partial y} \right) dx \, dy
$$

For a single skyrmion, $Q = \pm 1$. In the SAF, the two layers carry opposite
topological charges ($Q_{\text{top}} = -Q_{\text{bot}}$), leading to a net
topological charge of zero and hence no skyrmion Hall effect.

## 7. Skyrmion Profile (Initial Condition)

The initial Neel skyrmion profile wraps the 1-D domain-wall solution
into a circle of radius $R$:

$$
\theta(r) = 2\arctan\left(\exp\left(-\frac{r - R}{\Delta}\right)\right)
$$

$$
m_x = \sin\theta \, \cos\varphi, \quad
m_y = \sin\theta \, \sin\varphi, \quad
m_z = \cos\theta
$$

where $r$ is the distance from the skyrmion center, $R$ is the radius of
the $m_z = 0$ contour, $\Delta$ is the wall width, and $\varphi = \text{atan2}(y - y_0, x - x_0)$ is the azimuthal angle
(Neel helicity).

The top layer has core down ($m_z = -1$ at center). The bottom layer is its
full reversal, $\mathbf{m}_{\text{bot}} = -\mathbf{m}_{\text{top}}$: core up
and in-plane spins pointing inward. The DMI energy is unchanged under
$\mathbf{m} \to -\mathbf{m}$, so both layers keep the DMI-favoured
chirality, and the two walls are antiparallel everywhere, the ground state
of the antiferromagnetic RKKY coupling.

## 8. Time Integration

The explicit LLGS equation is integrated using the classical 4th-order
Runge-Kutta (RK4) method. At each substep, the spin configuration is
renormalized to enforce the unit-length constraint $|\mathbf{m}| = 1$:

$$
\mathbf{k}_1 = \Delta t \, f(\mathbf{m})
$$
$$
\mathbf{k}_2 = \Delta t \, f\!\left(\text{norm}\!\left(\mathbf{m} + \frac{\mathbf{k}_1}{2}\right)\right)
$$
$$
\mathbf{k}_3 = \Delta t \, f\!\left(\text{norm}\!\left(\mathbf{m} + \frac{\mathbf{k}_2}{2}\right)\right)
$$
$$
\mathbf{k}_4 = \Delta t \, f\!\left(\text{norm}\!\left(\mathbf{m} + \mathbf{k}_3\right)\right)
$$
$$
\mathbf{m}^{n+1} = \text{norm}\!\left(\mathbf{m}^n
+ \frac{\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4}{6}\right)
$$

Both SAF layers are evolved simultaneously, as the RKKY coupling makes them
interdependent.

At finite temperature a random thermal field $\mathbf{h}$ is added to
$\mathbf{H}_{\text{eff}}$. Each component in each cell is Gaussian white
noise with
$\langle h_i(t) h_j(t') \rangle = \sigma^2 \delta_{ij} \delta(t - t')$ and
$\sigma^2 = 2 \alpha k_B T / (\gamma M_s V_{\text{cell}})$, the strength the
fluctuation-dissipation theorem requires for the dynamics to relax to the
Boltzmann distribution. The equation is read in the Stratonovich sense and
integrated with the Heun predictor-corrector scheme, which reuses the same
noise sample in both stages.

## 9. Boundary Conditions

Periodic boundary conditions (PBC) are applied in both $x$ and $y$ directions
by default, implemented via `np.roll` on the spin arrays. Finite samples use
free edges instead, with the Rohart-Thiaville free-edge condition
$\partial_n \mathbf{m} = (D / 2A_{\text{ex}}) (\hat{\mathbf{n}} \times \hat{z}) \times \mathbf{m}$
at the boundary.

## 10. Simulation Parameters

All material parameters are taken from [1] for the optimized SAF stack
Pt(3)/Co(1.58)/Ru(0.85)/Pt(0.5)/Co(1.58)/Ru(0.85) (thicknesses in nm). The
magnetic thickness used for each Co layer is $t_{\text{Co}} = 1.3$ nm, and
the spacer separating the layers is $d_{\text{Ru}} = 1.35$ nm.

| Parameter | Symbol | Value | Unit |
|---|---|---|---|
| Saturation magnetization | $M_s$ | $1.43 \times 10^6$ | A/m |
| Exchange stiffness | $A_{\text{ex}}$ | $16 \times 10^{-12}$ | J/m |
| DMI constant | $D$ | $0.85 \times 10^{-3}$ | J/m$^2$ |
| Anisotropy (top layer) | $K_{\text{top}}$ | $1.3106 \times 10^6$ | J/m$^3$ |
| Anisotropy (bottom layer) | $K_{\text{bot}}$ | $1.31 \times 10^6$ | J/m$^3$ |
| Gilbert damping | $\alpha$ | 0.14 | -- |
| Co layer thickness | $t_{\text{Co}}$ | 1.3 | nm |
| Spacer thickness | $d_{\text{Ru}}$ | 1.35 | nm |
| Gyromagnetic ratio | $\gamma$ | $194.8 \times 10^9$ | rad/(s T) |
| RKKY coupling field | $\mu_0 H_{\text{RKKY}}$ | 205 | mT |
| DL-SOT coefficient | $\chi_{\text{DL}}$ | $2.21 \times 10^{-14}$ | T A$^{-1}$ m$^2$ |
| FL-SOT coefficient | $\chi_{\text{FL}}$ | $0.53 \times 10^{-14}$ | T A$^{-1}$ m$^2$ |
| Domain wall width | $\Delta$ | 27 | nm |

The paper reports $D = 0.62 \pm 0.24$ mJ/m$^2$; $0.85$ mJ/m$^2$ lies in the
upper part of that band. The measured top-layer anisotropy,
$1.294 \times 10^6$ J/m$^3$ ($\mu_0 H_k = 12.4$ mT), is raised to
$1.3106 \times 10^6$ J/m$^3$ ($\mu_0 H_k = 36$ mT), as in the paper's own
simulations, so the background domain is not reversed by the SOT at the
largest currents.

### Derived quantities

| Quantity | Formula | Value |
|---|---|---|
| $K_{\text{eff, top}}$ | $K_{\text{top}} - \mu_0 M_s^2 / 2$ | $\approx 2.6 \times 10^4$ J/m$^3$ |
| $K_{\text{eff, bot}}$ | $K_{\text{bot}} - \mu_0 M_s^2 / 2$ | $\approx 2.5 \times 10^4$ J/m$^3$ |
| Wall-width parameter | $\Delta = \sqrt{A_{\text{ex}} / K_{\text{eff}}}$ | $\approx$ 25 nm |
| Critical DMI | $D_c = (4/\pi)\sqrt{A_{\text{ex}} K_{\text{eff}}}$ | $\approx$ 0.82 mJ/m$^2$ |

### Numerical parameters

| Parameter | Value | Description |
|---|---|---|
| $n_x \times n_y$ | 256 x 256 | Lattice sites |
| $a$ | 2 nm | Lattice constant |
| $\Delta t$ | $5 \times 10^{-14}$ s | Time step |
| Relaxation steps | 10000 | 500 ps at $J = 0$ |
| Drive steps | 5000 | 250 ps with current |
| Current density | $J(t)$ | current pulse, set per run |
| Spin polarization | $\hat{\mathbf{p}} = \hat{y}$ | From spin Hall effect |
| Initial skyrmion radius | 93.25 nm | Relaxes to equilibrium |

### Stability condition

The RK4 time step must satisfy:

$$
\omega_{\text{max}} \cdot \Delta t < 2.8
$$

where $\omega_{\text{max}} = \gamma \cdot H_{\text{max}}$ and
$H_{\text{max}} \approx 4 C_{\text{ex}}$ is the largest exchange field on the
lattice, reached when neighbouring spins are antiparallel. With
$C_{\text{ex}} = 2A_{\text{ex}}/(M_s a^2) \approx 5.6$ T,
$H_{\text{max}} \approx 22$ T and $\omega_{\text{max}} \cdot \Delta t \approx 0.22$,
well within the RK4 stability region.

## 11. Output Format

Spin configurations are written in LAMMPS dump format. Since OVITO does not
natively support spin degrees of freedom, the three spin components
($m_x$, $m_y$, $m_z$) are written as force components (`fx`, `fy`, `fz`),
which OVITO can render as vector arrows. Positions are output in nanometers.

Type 1 corresponds to the top Co layer ($z = 0$), type 2 to the bottom Co
layer ($z = -t_{\text{Co}}$).

## References

[1] V. T. Pham, N. Sisodia, I. Di Manici, J. Urrestarazu-Larranaga et al.,
"Fast current-induced skyrmion motion in synthetic antiferromagnets,"
Nature Physics (2024).
