// Newell finite-prism demag kernel via mumax3-style variable-density
// Gauss-Legendre numerical integration of the surface-charge formulation.
// Mirrors src/simulator/demag_newell.py.
//
// Each lattice cell is treated as a uniformly magnetized rectangular
// prism (a, a, t_Co). The tensor for one source/dest cell pair is the
// dest-volume average of the field produced by the source's two
// charged faces (sigma = +/- M_u on the +/- u-faces). Integration
// density is adaptive: maxSize = edge_to_edge_distance / accuracy with
// one quadrature node per maxSize along each axis. Source-surface
// nodes are doubled (SURFACE_STAGGER = 2) along the in-face axes.
//
// The (0, 0) self-cell entry of the self-layer kernel uses Aharoni's
// (1998) closed form for the diagonals; off-diagonals vanish by cell
// mirror symmetry.
//
// A convergence assertion compares the kernel built at the requested
// `accuracy` against `2 * accuracy`; if the relative difference in
// any k-space component exceeds `tol_conv` a runtime_error is raised.
#include "skyrmion/demag.hpp"
#include "skyrmion/fft2d.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace skyrmion {

namespace {

constexpr Real kPi = 3.14159265358979323846;
constexpr int  kSurfaceStagger = 2;

// -------- Aharoni (1998) closed-form demag factor along the c-axis ---------
Real aharoni_demag_factor(Real a, Real b, Real c) {
    const Real ah = a / 2.0;
    const Real bh = b / 2.0;
    const Real ch = c / 2.0;
    const Real abc = std::sqrt(ah*ah + bh*bh + ch*ch);
    const Real ab  = std::sqrt(ah*ah + bh*bh);
    const Real ac  = std::sqrt(ah*ah + ch*ch);
    const Real bc  = std::sqrt(bh*bh + ch*ch);
    return (1.0 / kPi) * (
          (bh*bh - ch*ch) / (2.0 * bh * ch) * std::log((abc - ah) / (abc + ah))
        + (ah*ah - ch*ch) / (2.0 * ah * ch) * std::log((abc - bh) / (abc + bh))
        + bh / (2.0 * ch) * std::log((ab + ah) / (ab - ah))
        + ah / (2.0 * ch) * std::log((ab + bh) / (ab - bh))
        + ch / (2.0 * ah) * std::log((bc - bh) / (bc + bh))
        + ch / (2.0 * bh) * std::log((ac - ah) / (ac + ah))
        + 2.0 * std::atan2(ah * bh, ch * abc)
        + (ah*ah*ah + bh*bh*bh - 2.0 * ch*ch*ch) / (3.0 * ah * bh * ch)
        + (ah*ah + bh*bh - 2.0 * ch*ch) / (3.0 * ah * bh * ch) * abc
        + ch / (ah * bh) * (ac + bc)
        - std::pow(ah*ah + ch*ch, 1.5) / (3.0 * ah * bh * ch)
        - std::pow(bh*bh + ch*ch, 1.5) / (3.0 * ah * bh * ch)
        - std::pow(ah*ah + bh*bh, 1.5) / (3.0 * ah * bh * ch));
}

// -------- Gauss-Legendre nodes/weights ------------------------------------
// nodes in [-1, +1], weights normalised so sum(weights) == 1
// (matches Python `_gl_unit_interval_nodes`).
void compute_gl_nodes(int n,
                      std::vector<Real>& nodes,
                      std::vector<Real>& weights) {
    if (n < 1) throw std::runtime_error("compute_gl_nodes: n must be >= 1.");
    nodes.assign(n, 0.0);
    weights.assign(n, 0.0);
    if (n == 1) {
        nodes[0] = 0.0;
        weights[0] = 1.0;
        return;
    }
    for (int i = 0; i < n; ++i) {
        // Tricomi initial guess for the (i+1)-th root.
        Real x = std::cos(kPi * (4.0 * (i + 1) - 1.0) / (4.0 * n + 2.0));
        Real Pn = 0.0, Pn_prime = 0.0;
        for (int it = 0; it < 200; ++it) {
            Real pm1 = 0.0;   // P_{k-1}
            Real p0  = 1.0;   // P_0
            Real p1  = 0.0;
            for (int k = 1; k <= n; ++k) {
                p1 = ((2.0 * k - 1.0) * x * p0 - (k - 1.0) * pm1) / k;
                pm1 = p0;
                p0 = p1;
            }
            Pn = p0;
            Pn_prime = n * (x * Pn - pm1) / (x * x - 1.0);
            const Real dx = Pn / Pn_prime;
            x -= dx;
            if (std::abs(dx) < 1e-15) break;
        }
        nodes[i] = x;
        // Final evaluation of P_n' at the converged x.
        Real pm1 = 0.0, p0 = 1.0, p1 = 0.0;
        for (int k = 1; k <= n; ++k) {
            p1 = ((2.0 * k - 1.0) * x * p0 - (k - 1.0) * pm1) / k;
            pm1 = p0;
            p0 = p1;
        }
        Pn_prime = n * (x * p0 - pm1) / (x * x - 1.0);
        weights[i] = 2.0 / ((1.0 - x * x) * Pn_prime * Pn_prime);
        weights[i] *= 0.5;   // normalise sum to 1
    }
}

struct GLEntry { std::vector<Real> nodes; std::vector<Real> weights; };

// Thread-local cache: build once per n the first time a thread asks.
const GLEntry& get_gl(int n) {
    static thread_local std::vector<GLEntry> cache;
    if (static_cast<int>(cache.size()) <= n) cache.resize(n + 1);
    GLEntry& e = cache[n];
    if (e.nodes.empty()) compute_gl_nodes(n, e.nodes, e.weights);
    return e;
}

// -------- Closest edge-to-edge distance in lattice units ------------------
inline Real delta_lat(int idx) {
    int v = std::abs(idx);
    if (v > 0) v -= 1;
    return static_cast<Real>(v);
}

// -------- 3x3 demag tensor for one cell pair ------------------------------
// Vectorised in Python via rank-5 broadcasting; here written as explicit
// nested loops so the hot path stays cache-friendly with stack-resident
// pole/displacement vectors.
void compute_one_pair_tensor(Real X, Real Y, Real Z,
                             const Real cs[3],
                             const int n_density[3],
                             Real out[3][3]) {
    const GLEntry& gx = get_gl(n_density[0]);
    const GLEntry& gy = get_gl(n_density[1]);
    const GLEntry& gz = get_gl(n_density[2]);
    const Real* rx = gx.nodes.data();  const Real* wrx = gx.weights.data();
    const Real* ry = gy.nodes.data();  const Real* wry = gy.weights.data();
    const Real* rz = gz.nodes.data();  const Real* wrz = gz.weights.data();
    const int nxv = n_density[0], nyv = n_density[1], nzv = n_density[2];
    // Pre-scale node positions to physical offsets [-cs/2, +cs/2].
    Real rx_off[64], ry_off[64], rz_off[64];
    for (int i = 0; i < nxv; ++i) rx_off[i] = rx[i] * cs[0] / 2.0;
    for (int i = 0; i < nyv; ++i) ry_off[i] = ry[i] * cs[1] / 2.0;
    for (int i = 0; i < nzv; ++i) rz_off[i] = rz[i] * cs[2] / 2.0;

    for (int u = 0; u < 3; ++u) for (int v = 0; v < 3; ++v) out[u][v] = 0.0;

    for (int u = 0; u < 3; ++u) {
        const int v = (u + 1) % 3;
        const int w = (u + 2) % 3;
        const int n_v = n_density[v] * kSurfaceStagger;
        const int n_w = n_density[w] * kSurfaceStagger;
        const GLEntry& gv = get_gl(n_v);
        const GLEntry& gw = get_gl(n_w);
        const Real* pv = gv.nodes.data();   const Real* wv = gv.weights.data();
        const Real* pw = gw.nodes.data();   const Real* ww = gw.weights.data();
        // Physical offsets along the in-face axes.
        Real pv_off[64], pw_off[64];
        for (int i = 0; i < n_v; ++i) pv_off[i] = pv[i] * cs[v] / 2.0;
        for (int i = 0; i < n_w; ++i) pw_off[i] = pw[i] * cs[w] / 2.0;
        const Real surface = cs[v] * cs[w];
        const Real prefactor = surface / (4.0 * kPi);
        Real Hx_sum = 0.0, Hy_sum = 0.0, Hz_sum = 0.0;
        for (int iv = 0; iv < n_v; ++iv) {
            const Real PV_pos = pv_off[iv];
            const Real Wv_iv = wv[iv];
            for (int iw = 0; iw < n_w; ++iw) {
                const Real PW_pos = pw_off[iw];
                const Real Wsurf = Wv_iv * ww[iw];
                // Pole positions: +M face at +cs[u]/2 (sigma=+1), -M at -cs[u]/2.
                Real pole_p[3], pole_m[3];
                pole_p[u] = +cs[u] / 2.0;
                pole_m[u] = -cs[u] / 2.0;
                pole_p[v] = PV_pos;  pole_m[v] = PV_pos;
                pole_p[w] = PW_pos;  pole_m[w] = PW_pos;
                for (int ix = 0; ix < nxv; ++ix) {
                    const Real r_dx = X + rx_off[ix];
                    const Real Wx = wrx[ix];
                    for (int iy = 0; iy < nyv; ++iy) {
                        const Real r_dy = Y + ry_off[iy];
                        const Real Wxy = Wx * wry[iy];
                        for (int iz = 0; iz < nzv; ++iz) {
                            const Real r_dz = Z + rz_off[iz];
                            const Real W_total = Wxy * wrz[iz] * Wsurf;
                            // + pole displacement
                            const Real dxp = r_dx - pole_p[0];
                            const Real dyp = r_dy - pole_p[1];
                            const Real dzp = r_dz - pole_p[2];
                            const Real r2p = dxp*dxp + dyp*dyp + dzp*dzp;
                            const Real fp  = W_total * prefactor
                                             / (r2p * std::sqrt(r2p));
                            // - pole displacement
                            const Real dxn = r_dx - pole_m[0];
                            const Real dyn = r_dy - pole_m[1];
                            const Real dzn = r_dz - pole_m[2];
                            const Real r2n = dxn*dxn + dyn*dyn + dzn*dzn;
                            const Real fn  = W_total * prefactor
                                             / (r2n * std::sqrt(r2n));
                            Hx_sum += dxp * fp - dxn * fn;
                            Hy_sum += dyp * fp - dyn * fn;
                            Hz_sum += dzp * fp - dzn * fn;
                        }
                    }
                }
            }
        }
        out[u][0] = Hx_sum;
        out[u][1] = Hy_sum;
        out[u][2] = Hz_sum;
    }
}

// -------- Closed-form Newell-Williams-Dunlop 1993 tensor ------------------
// Exact analytic alternative to the Gauss-Legendre quadrature above. Direct
// transcription of OOMMF's Oxs_Newell_f / Oxs_Newell_g antiderivatives and
// the Python _newell_tensor_closed. The 27-corner second-difference of f
// (diagonals) / g (off-diagonals) on the (-1,0,+1)^3 lattice gives the
// tensor between two prisms of dims (dx,dy,dz) at offset (X,Y,Z), in the
// depolarizing-positive sign convention (H = -mu0*Ms*N*m, no later negation).

Real newell_f(Real x, Real y, Real z) {
    x = std::abs(x); y = std::abs(y); z = std::abs(z);
    const Real xsq = x * x, ysq = y * y, zsq = z * z;
    const Real R2 = xsq + ysq + zsq;
    if (R2 <= 0.0) return 0.0;
    const Real R = std::sqrt(R2);
    Real sum = 0.0;
    if (z > 0.0) {
        sum += 2.0 * (2.0 * xsq - ysq - zsq) * R;
        const Real t1 = x * y * z;
        if (t1 > 0.0) sum += -12.0 * t1 * std::atan2(y * z, x * R);
        const Real t2 = xsq + zsq;
        if (y > 0.0 && t2 > 0.0)
            sum += 3.0 * y * (zsq - xsq) * std::log(((y + R) * (y + R)) / t2);
        const Real t3 = xsq + ysq;
        if (t3 > 0.0)
            sum += 3.0 * z * (ysq - xsq) * std::log(((z + R) * (z + R)) / t3);
    } else {
        if (x == y) {
            const Real K = 2.0 * std::sqrt(2.0)
                           - 6.0 * std::log(1.0 + std::sqrt(2.0));
            sum += K * xsq * x;
        } else {
            sum += 2.0 * (2.0 * xsq - ysq) * R;
            if (y > 0.0 && x > 0.0)
                sum += -6.0 * y * xsq * std::log((y + R) / x);
        }
    }
    return sum / 12.0;
}

Real newell_g(Real x, Real y, Real z) {
    Real result_sign = 1.0;
    if (x < 0.0) result_sign *= -1.0;
    if (y < 0.0) result_sign *= -1.0;
    x = std::abs(x); y = std::abs(y); z = std::abs(z);
    const Real xsq = x * x, ysq = y * y, zsq = z * z;
    const Real R2 = xsq + ysq + zsq;
    if (R2 <= 0.0) return 0.0;
    const Real R = std::sqrt(R2);
    Real sum = -2.0 * x * y * R;
    if (z > 0.0) {
        sum += -z * zsq * std::atan2(x * y, z * R);
        sum += -3.0 * z * ysq * std::atan2(x * z, y * R);
        sum += -3.0 * z * xsq * std::atan2(y * z, x * R);
        const Real t1 = xsq + ysq;
        if (t1 > 0.0)
            sum += 3.0 * x * y * z * std::log(((z + R) * (z + R)) / t1);
        const Real t2 = ysq + zsq;
        if (t2 > 0.0)
            sum += 0.5 * y * (3.0 * zsq - ysq)
                   * std::log(((x + R) * (x + R)) / t2);
        const Real t3 = xsq + zsq;
        if (t3 > 0.0)
            sum += 0.5 * x * (3.0 * zsq - xsq)
                   * std::log(((y + R) * (y + R)) / t3);
    } else {
        if (y > 0.0) sum += -y * ysq * std::log((x + R) / y);
        if (x > 0.0) sum += -x * xsq * std::log((y + R) / x);
    }
    return result_sign * sum / 6.0;
}

template <typename Antideriv>
Real newell_27_corner(Real X, Real Y, Real Z, Real dx, Real dy, Real dz,
                      Antideriv f_func) {
    static const Real w[3] = {-1.0, 2.0, -1.0};
    static const int off[3] = {-1, 0, 1};
    Real val = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                val += w[i] * w[j] * w[k]
                       * f_func(X + off[i] * dx, Y + off[j] * dy,
                                Z + off[k] * dz);
            }
        }
    }
    return val / (4.0 * kPi * dx * dy * dz);
}

// Point-dipole far-field tensor: N_ab = V/(4 pi r^3)(d_ab - 3 rh_a rh_b).
// Depolarizing-positive convention; relative error O((cell/r)^2).
void dipole_tensor(Real X, Real Y, Real Z, Real dx, Real dy, Real dz,
                   Real out[3][3]) {
    const Real r2 = X * X + Y * Y + Z * Z;
    if (r2 == 0.0) {
        throw std::runtime_error(
            "dipole_tensor: zero separation; the dipole asymptote is "
            "undefined at the self-cell.");
    }
    const Real r = std::sqrt(r2);
    const Real pref = dx * dy * dz / (4.0 * kPi * r2 * r);
    const Real rh[3] = {X / r, Y / r, Z / r};
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            const Real delta = (a == b) ? 1.0 : 0.0;
            out[a][b] = pref * (delta - 3.0 * rh[a] * rh[b]);
        }
    }
}

void newell_tensor_closed(Real X, Real Y, Real Z, Real dx, Real dy, Real dz,
                          Real out[3][3]) {
    out[0][0] = newell_27_corner(X, Y, Z, dx, dy, dz, newell_f);
    out[1][1] = newell_27_corner(Y, X, Z, dy, dx, dz, newell_f);
    out[2][2] = newell_27_corner(Z, Y, X, dz, dy, dx, newell_f);
    out[0][1] = newell_27_corner(X, Y, Z, dx, dy, dz, newell_g);
    out[1][0] = out[0][1];
    out[0][2] = newell_27_corner(X, Z, Y, dx, dz, dy, newell_g);
    out[2][0] = out[0][2];
    out[1][2] = newell_27_corner(Y, Z, X, dy, dz, dx, newell_g);
    out[2][1] = out[1][2];
}

struct LayerPairKernel {
    int ny, nx;
    std::vector<Real> Nxx, Nyy, Nzz, Nxy, Nxz, Nyz;  // (ny * nx) real-space
};

// images_x/images_y: periodic image wraps per direction (0 = min image).
// Image terms always use the closed/dipole hybrid (far-field accurate).
LayerPairKernel build_layer_pair_kernel(int nx, int ny, Real dx, Real dy,
                                        Real t_layer, Real Z_separation,
                                        Real accuracy, DemagMethod method,
                                        int images_x, int images_y) {
    const Real cs[3] = {dx, dy, t_layer};
    const Real L = std::min({dx, dy, t_layer});
    LayerPairKernel K;
    K.ny = ny; K.nx = nx;
    const std::size_t n_tot = static_cast<std::size_t>(ny) * nx;
    K.Nxx.assign(n_tot, 0.0);  K.Nyy.assign(n_tot, 0.0);  K.Nzz.assign(n_tot, 0.0);
    K.Nxy.assign(n_tot, 0.0);  K.Nxz.assign(n_tot, 0.0);  K.Nyz.assign(n_tot, 0.0);

    auto signed_idx = [](int idx, int n) {
        return (idx <= n / 2) ? idx : (idx - n);
    };

    // Closed form already returns the depolarizing-positive tensor; the
    // quadrature returns mumax3's +H/M convention (= -N), so it is negated
    // into the same convention as it is stored.
    const Real sign = (method == DemagMethod::Closed) ? 1.0 : -1.0;

    // Far-field crossover radius: 40 cells.
    const Real r_c = 40.0 * std::max({dx, dy, t_layer});

#ifdef SKYRMION_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int j_idx = 0; j_idx < ny; ++j_idx) {
        const int iy_s = signed_idx(j_idx, ny);
        for (int i_idx = 0; i_idx < nx; ++i_idx) {
            const int ix_s = signed_idx(i_idx, nx);
            const Real X = ix_s * dx;
            const Real Y = iy_s * dy;
            const Real Z = Z_separation;
            Real tensor[3][3];
            if (method == DemagMethod::Closed) {
                // Beyond r_c the 27-corner difference cancels catastrophically.
                if (std::sqrt(X * X + Y * Y + Z * Z) > r_c) {
                    // Dipole asymptote: O((cell/r)^2) relative error.
                    dipole_tensor(X, Y, Z, dx, dy, t_layer, tensor);
                } else {
                    newell_tensor_closed(X, Y, Z, dx, dy, t_layer, tensor);
                }
            } else {
                const Real dxe = delta_lat(ix_s) * dx;
                const Real dye = delta_lat(iy_s) * dy;
                Real dze;
                if (Z_separation == 0.0) dze = 0.0;
                else dze = std::max(std::abs(Z_separation) - t_layer, 0.0);
                Real d = std::sqrt(dxe*dxe + dye*dye + dze*dze);
                if (d == 0.0) d = L;
                const Real maxSize = d / accuracy;
                const int n_x =
                    std::max(static_cast<int>(dx / maxSize + 0.5), 1);
                const int n_y =
                    std::max(static_cast<int>(dy / maxSize + 0.5), 1);
                const int n_z =
                    std::max(static_cast<int>(t_layer / maxSize + 0.5), 1);
                const int n_density[3] = {n_x, n_y, n_z};
                compute_one_pair_tensor(X, Y, Z, cs, n_density, tensor);
            }
            const std::size_t idx = static_cast<std::size_t>(j_idx) * nx + i_idx;
            K.Nxx[idx] = sign * tensor[0][0];
            K.Nyy[idx] = sign * tensor[1][1];
            K.Nzz[idx] = sign * tensor[2][2];
            K.Nxy[idx] = sign * tensor[0][1];
            K.Nxz[idx] = sign * tensor[0][2];
            K.Nyz[idx] = sign * tensor[1][2];
        }
    }

    // Aharoni override at (0, 0) for the self-layer kernel: the
    // quadrature is singular near the source-coincides-dest pole, and the
    // closed form is overridden too for exact parity with the Python path.
    if (Z_separation == 0.0) {
        K.Nxx[0] = aharoni_demag_factor(dy, t_layer, dx);
        K.Nyy[0] = aharoni_demag_factor(dx, t_layer, dy);
        K.Nzz[0] = aharoni_demag_factor(dx, dy, t_layer);
        K.Nxy[0] = 0.0;
        K.Nxz[0] = 0.0;
        K.Nyz[0] = 0.0;
    }

    // Periodic image sum K(off) += sum_{w != 0} N(off + w L); after the
    // Aharoni override so (0, 0) keeps the exact isolated self-term.
    if (images_x > 0 || images_y > 0) {
        const Real r_c_img = 40.0 * std::max({dx, dy, t_layer});
#ifdef SKYRMION_OPENMP
        #pragma omp parallel for schedule(dynamic)
#endif
        for (int j_idx = 0; j_idx < ny; ++j_idx) {
            const int iy_s = signed_idx(j_idx, ny);
            for (int i_idx = 0; i_idx < nx; ++i_idx) {
                const int ix_s = signed_idx(i_idx, nx);
                const std::size_t idx =
                    static_cast<std::size_t>(j_idx) * nx + i_idx;
                for (int wy = -images_y; wy <= images_y; ++wy) {
                    for (int wx = -images_x; wx <= images_x; ++wx) {
                        if (wx == 0 && wy == 0) continue;
                        const Real X = (ix_s + wx * nx) * dx;
                        const Real Y = (iy_s + wy * ny) * dy;
                        Real tensor[3][3];
                        if (std::sqrt(X * X + Y * Y
                                      + Z_separation * Z_separation)
                            > r_c_img) {
                            dipole_tensor(X, Y, Z_separation,
                                          dx, dy, t_layer, tensor);
                        } else {
                            newell_tensor_closed(X, Y, Z_separation,
                                                 dx, dy, t_layer, tensor);
                        }
                        K.Nxx[idx] += tensor[0][0];
                        K.Nyy[idx] += tensor[1][1];
                        K.Nzz[idx] += tensor[2][2];
                        K.Nxy[idx] += tensor[0][1];
                        K.Nxz[idx] += tensor[0][2];
                        K.Nyz[idx] += tensor[1][2];
                    }
                }
            }
        }
    }
    return K;
}

// FFT a real-space (ny, nx) kernel to (ny, nx) complex full-spectrum
// with 1/(ny*nx) normalisation baked in so the runtime c2c inverse
// transform needs no post-scaling.
void fft_real_kernel(const std::vector<Real>& real_kernel,
                     int ny, int nx, FFT2D& fft,
                     std::vector<Complex>& out) {
    const Real norm = 1.0 / (static_cast<Real>(ny) * nx);
    fftw_complex* in = fft.scratch_in();
    const std::size_t n = static_cast<std::size_t>(ny) * nx;
    for (std::size_t k = 0; k < n; ++k) {
        in[k][0] = real_kernel[k];
        in[k][1] = 0.0;
    }
    fft.execute_fwd();
    const fftw_complex* src = fft.scratch_out();
    out.assign(n, Complex{0, 0});
    for (std::size_t k = 0; k < n; ++k) {
        out[k] = Complex{src[k][0] * norm, src[k][1] * norm};
    }
}

// Build the k-space kernel on a (gny, gnx) FFT grid. For the periodic
// kinds gny/gnx == p.ny/p.nx; for free-BC they are the doubled (2*phys)
// grid and `freebc` records the physical size for pad/crop at apply time.
DemagKernels assemble_kernel_dict(const Params& p, Real accuracy,
                                  int gny, int gnx, bool freebc,
                                  int images_x, int images_y) {
    LayerPairKernel self_k = build_layer_pair_kernel(
        gnx, gny, p.a, p.a, p.t_Co, 0.0, accuracy, p.demag_method,
        images_x, images_y);
    LayerPairKernel inter_k = build_layer_pair_kernel(
        gnx, gny, p.a, p.a, p.t_Co, p.t_Co + p.d_Ru, accuracy,
        p.demag_method, images_x, images_y);

    FFT2D fft(gny, gnx, 0);
    DemagKernels K;
    K.ny = gny; K.nx = gnx;
    K.freebc = freebc;
    K.ny_phys = freebc ? p.ny : gny;
    K.nx_phys = freebc ? p.nx : gnx;
    K.mu0_Ms = p.mu0 * p.Ms;
    K.t_Co = p.t_Co;
    K.d_Ru = p.d_Ru;

    fft_real_kernel(self_k.Nxx, gny, gnx, fft, K.Nxx_self);
    fft_real_kernel(self_k.Nyy, gny, gnx, fft, K.Nyy_self);
    fft_real_kernel(self_k.Nzz, gny, gnx, fft, K.Nzz_self);
    fft_real_kernel(self_k.Nxy, gny, gnx, fft, K.Nxy_self);
    fft_real_kernel(inter_k.Nxx, gny, gnx, fft, K.Nxx_inter);
    fft_real_kernel(inter_k.Nyy, gny, gnx, fft, K.Nyy_inter);
    fft_real_kernel(inter_k.Nzz, gny, gnx, fft, K.Nzz_inter);
    fft_real_kernel(inter_k.Nxy, gny, gnx, fft, K.Nxy_inter);
    fft_real_kernel(inter_k.Nxz, gny, gnx, fft, K.Nxz_inter);
    fft_real_kernel(inter_k.Nyz, gny, gnx, fft, K.Nyz_inter);
    return K;
}

void check_convergence(const char* name,
                       const std::vector<Complex>& lo,
                       const std::vector<Complex>& hi,
                       Real tol_conv) {
    Real max_hi = 0.0, max_diff = 0.0;
    for (std::size_t k = 0; k < lo.size(); ++k) {
        const Real ah = std::abs(hi[k]);
        const Real ad = std::abs(hi[k] - lo[k]);
        if (ah > max_hi)   max_hi  = ah;
        if (ad > max_diff) max_diff = ad;
    }
    if (max_hi < 1e-30) return;
    const Real rel_err = max_diff / max_hi;
    if (rel_err > tol_conv) {
        throw std::runtime_error(
            std::string("demag_newell: convergence failed for component ")
            + name + " (rel_err=" + std::to_string(rel_err)
            + " > tol_conv=" + std::to_string(tol_conv) + ").");
    }
}

void check_all_components(const DemagKernels& lo, const DemagKernels& hi,
                          Real tol_conv) {
    check_convergence("Nxx_self",  lo.Nxx_self,  hi.Nxx_self,  tol_conv);
    check_convergence("Nyy_self",  lo.Nyy_self,  hi.Nyy_self,  tol_conv);
    check_convergence("Nzz_self",  lo.Nzz_self,  hi.Nzz_self,  tol_conv);
    check_convergence("Nxy_self",  lo.Nxy_self,  hi.Nxy_self,  tol_conv);
    check_convergence("Nxx_inter", lo.Nxx_inter, hi.Nxx_inter, tol_conv);
    check_convergence("Nyy_inter", lo.Nyy_inter, hi.Nyy_inter, tol_conv);
    check_convergence("Nzz_inter", lo.Nzz_inter, hi.Nzz_inter, tol_conv);
    check_convergence("Nxy_inter", lo.Nxy_inter, hi.Nxy_inter, tol_conv);
    check_convergence("Nxz_inter", lo.Nxz_inter, hi.Nxz_inter, tol_conv);
    check_convergence("Nyz_inter", lo.Nyz_inter, hi.Nyz_inter, tol_conv);
}

void validate_newell_params(const Params& p, Real accuracy, Real tol_conv) {
    if (accuracy <= 0.0) {
        throw std::runtime_error("demag_newell: accuracy must be positive.");
    }
    if (tol_conv <= 0.0 || tol_conv >= 1.0) {
        throw std::runtime_error("demag_newell: tol_conv must lie in (0, 1).");
    }
    if (p.t_Co <= 0.0) {
        throw std::runtime_error("demag_newell: t_Co must be positive.");
    }
    if (p.d_Ru < 0.0) {
        throw std::runtime_error("demag_newell: d_Ru must be non-negative.");
    }
    if (p.nx < 2 || p.ny < 2) {
        throw std::runtime_error("demag_newell: nx, ny must be >= 2.");
    }
}

} // namespace

DemagKernels precompute_demag_newell(const Params& p,
                                     Real accuracy, Real tol_conv) {
    validate_newell_params(p, accuracy, tol_conv);
    // Closed form is exact and accuracy-independent: build once, no check.
    // Doubly periodic: image sums along both directions.
    const int w = p.pbc_images;
    if (p.demag_method == DemagMethod::Closed) {
        return assemble_kernel_dict(p, accuracy, p.ny, p.nx, false, w, w);
    }
    DemagKernels K_lo = assemble_kernel_dict(p, accuracy, p.ny, p.nx, false,
                                             w, w);
    DemagKernels K_hi = assemble_kernel_dict(p, 2.0 * accuracy,
                                             p.ny, p.nx, false, w, w);
    check_all_components(K_lo, K_hi, tol_conv);
    return K_hi;
}

DemagKernels precompute_demag_newell_freebc(const Params& p,
                                            Real accuracy, Real tol_conv) {
    validate_newell_params(p, accuracy, tol_conv);
    // 2N zero-padded grid -> isolated (no periodic image) convolution.
    const int gny = 2 * p.ny, gnx = 2 * p.nx;
    // Isolated (free) boundary: no periodic images.
    if (p.demag_method == DemagMethod::Closed) {
        return assemble_kernel_dict(p, accuracy, gny, gnx, true, 0, 0);
    }
    DemagKernels K_lo = assemble_kernel_dict(p, accuracy, gny, gnx, true,
                                             0, 0);
    DemagKernels K_hi = assemble_kernel_dict(p, 2.0 * accuracy,
                                             gny, gnx, true, 0, 0);
    check_all_components(K_lo, K_hi, tol_conv);
    return K_hi;
}

DemagKernels precompute_demag_racetrack(const Params& p,
                                              Real accuracy, Real tol_conv) {
    validate_newell_params(p, accuracy, tol_conv);
    // Mixed track BC: periodic along x (grid width nx), free/isolated
    // along y (grid height 2*ny, zero-padded). The freebc flag drives the
    // top-left pad/crop in DemagState; since gnx == nx the field fills all
    // columns (circular x) while the doubled rows give linear-conv y.
    const int gny = 2 * p.ny, gnx = p.nx;
    // Images along the periodic x direction only; y is free (padded).
    const int w = p.pbc_images;
    if (p.demag_method == DemagMethod::Closed) {
        return assemble_kernel_dict(p, accuracy, gny, gnx, true, w, 0);
    }
    DemagKernels K_lo = assemble_kernel_dict(p, accuracy, gny, gnx, true,
                                             w, 0);
    DemagKernels K_hi = assemble_kernel_dict(p, 2.0 * accuracy,
                                             gny, gnx, true, w, 0);
    check_all_components(K_lo, K_hi, tol_conv);
    return K_hi;
}

} // namespace skyrmion
