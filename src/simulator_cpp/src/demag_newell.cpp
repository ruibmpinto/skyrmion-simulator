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

struct LayerPairKernel {
    int ny, nx;
    std::vector<Real> Nxx, Nyy, Nzz, Nxy, Nxz, Nyz;  // (ny * nx) real-space
};

LayerPairKernel build_layer_pair_kernel(int nx, int ny, Real dx, Real dy,
                                        Real t_layer, Real Z_separation,
                                        Real accuracy) {
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
            const Real dxe = delta_lat(ix_s) * dx;
            const Real dye = delta_lat(iy_s) * dy;
            Real dze;
            if (Z_separation == 0.0) dze = 0.0;
            else dze = std::max(std::abs(Z_separation) - t_layer, 0.0);
            Real d = std::sqrt(dxe*dxe + dye*dye + dze*dze);
            if (d == 0.0) d = L;
            const Real maxSize = d / accuracy;
            const int n_x = std::max(static_cast<int>(dx / maxSize + 0.5), 1);
            const int n_y = std::max(static_cast<int>(dy / maxSize + 0.5), 1);
            const int n_z = std::max(static_cast<int>(t_layer / maxSize + 0.5), 1);
            const int n_density[3] = {n_x, n_y, n_z};
            Real tensor[3][3];
            compute_one_pair_tensor(X, Y, Z, cs, n_density, tensor);
            const std::size_t idx = static_cast<std::size_t>(j_idx) * nx + i_idx;
            K.Nxx[idx] = tensor[0][0];
            K.Nyy[idx] = tensor[1][1];
            K.Nzz[idx] = tensor[2][2];
            K.Nxy[idx] = tensor[0][1];
            K.Nxz[idx] = tensor[0][2];
            K.Nyz[idx] = tensor[1][2];
        }
    }

    // Sign convention: mumax3 stores +H/M; slab uses H = -mu0*Ms*N*m.
    for (std::size_t k = 0; k < n_tot; ++k) {
        K.Nxx[k] = -K.Nxx[k];
        K.Nyy[k] = -K.Nyy[k];
        K.Nzz[k] = -K.Nzz[k];
        K.Nxy[k] = -K.Nxy[k];
        K.Nxz[k] = -K.Nxz[k];
        K.Nyz[k] = -K.Nyz[k];
    }

    // Aharoni override at (0, 0) for the self-layer kernel: the
    // quadrature is singular near the source-coincides-dest pole.
    if (Z_separation == 0.0) {
        K.Nxx[0] = aharoni_demag_factor(dy, t_layer, dx);
        K.Nyy[0] = aharoni_demag_factor(dx, t_layer, dy);
        K.Nzz[0] = aharoni_demag_factor(dx, dy, t_layer);
        K.Nxy[0] = 0.0;
        K.Nxz[0] = 0.0;
        K.Nyz[0] = 0.0;
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

DemagKernels assemble_kernel_dict(const Params& p, Real accuracy) {
    LayerPairKernel self_k = build_layer_pair_kernel(
        p.nx, p.ny, p.a, p.a, p.t_Co, 0.0, accuracy);
    LayerPairKernel inter_k = build_layer_pair_kernel(
        p.nx, p.ny, p.a, p.a, p.t_Co, p.t_Co + p.d_Ru, accuracy);

    FFT2D fft(p.ny, p.nx, 0);
    DemagKernels K;
    K.ny = p.ny; K.nx = p.nx;
    K.mu0_Ms = p.mu0 * p.Ms;
    K.t_Co = p.t_Co;
    K.d_Ru = p.d_Ru;

    fft_real_kernel(self_k.Nxx, p.ny, p.nx, fft, K.Nxx_self);
    fft_real_kernel(self_k.Nyy, p.ny, p.nx, fft, K.Nyy_self);
    fft_real_kernel(self_k.Nzz, p.ny, p.nx, fft, K.Nzz_self);
    fft_real_kernel(self_k.Nxy, p.ny, p.nx, fft, K.Nxy_self);
    fft_real_kernel(inter_k.Nxx, p.ny, p.nx, fft, K.Nxx_inter);
    fft_real_kernel(inter_k.Nyy, p.ny, p.nx, fft, K.Nyy_inter);
    fft_real_kernel(inter_k.Nzz, p.ny, p.nx, fft, K.Nzz_inter);
    fft_real_kernel(inter_k.Nxy, p.ny, p.nx, fft, K.Nxy_inter);
    fft_real_kernel(inter_k.Nxz, p.ny, p.nx, fft, K.Nxz_inter);
    fft_real_kernel(inter_k.Nyz, p.ny, p.nx, fft, K.Nyz_inter);
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

} // namespace

DemagKernels precompute_demag_newell(const Params& p,
                                     Real accuracy, Real tol_conv) {
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
    DemagKernels K_lo = assemble_kernel_dict(p, accuracy);
    DemagKernels K_hi = assemble_kernel_dict(p, 2.0 * accuracy);
    check_convergence("Nxx_self",  K_lo.Nxx_self,  K_hi.Nxx_self,  tol_conv);
    check_convergence("Nyy_self",  K_lo.Nyy_self,  K_hi.Nyy_self,  tol_conv);
    check_convergence("Nzz_self",  K_lo.Nzz_self,  K_hi.Nzz_self,  tol_conv);
    check_convergence("Nxy_self",  K_lo.Nxy_self,  K_hi.Nxy_self,  tol_conv);
    check_convergence("Nxx_inter", K_lo.Nxx_inter, K_hi.Nxx_inter, tol_conv);
    check_convergence("Nyy_inter", K_lo.Nyy_inter, K_hi.Nyy_inter, tol_conv);
    check_convergence("Nzz_inter", K_lo.Nzz_inter, K_hi.Nzz_inter, tol_conv);
    check_convergence("Nxy_inter", K_lo.Nxy_inter, K_hi.Nxy_inter, tol_conv);
    check_convergence("Nxz_inter", K_lo.Nxz_inter, K_hi.Nxz_inter, tol_conv);
    check_convergence("Nyz_inter", K_lo.Nyz_inter, K_hi.Nyz_inter, tol_conv);
    return K_hi;
}

} // namespace skyrmion
