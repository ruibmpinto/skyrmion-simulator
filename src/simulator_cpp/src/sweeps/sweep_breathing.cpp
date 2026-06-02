// Skyrmion breathing-mode test: relax to equilibrium, apply a small
// radial m_z perturbation, then watch the free LLG dynamics ring back.
// Port of scripts/sweep_breathing.py (local-K_eff path). Custom IC, so
// this binary drives sweep::run_trace directly rather than run_point.
#include "skyrmion/integrator.hpp"
#include "skyrmion/sweep/sweep_common.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace skyrmion;
using namespace skyrmion::sweep;

namespace {

// m_z *= (1 - eps), then renormalize. Expands the m_z=0 ring slightly
// — a near-pure radial breathing perturbation.
void radial_perturb(Field3& m, double eps) {
    const int ny = m.ny, nx = m.nx;
    for (int i = 0; i < ny; ++i)
        for (int j = 0; j < nx; ++j)
            m(i, j, 2) *= (1.0 - eps);
    normalize_inplace(m);
}

} // namespace

int main() {
    // ----- Run configuration -------------------------------------------------
    const std::vector<double> D_values = {
        0.62e-3, 0.72e-3, 0.80e-3, 0.85e-3, 0.90e-3, 0.95e-3, 1.00e-3};
    const double perturb_eps = 0.03;
    const double free_alpha = 0.14;       // paper Set A, free evolution
    const double relax_alpha = 1.0;       // over-damped quench
    const double free_time = 1.0e-9;
    const int nx = 256, ny = 256;
    const double dt = 5.0e-14;
    const double sample_dt = 2.0e-12;
    const int relax_max_steps = 200000;
    const double relax_tol_torque = 1.0e-5;
    const double relax_tol_dE = 1.0e-8;
    const int relax_check_every = 1000;
    const bool dump_snapshots = true;
    const double snapshot_dt = 12.5e-12;         // was 25.0e-12
    const std::string out_dir = "output/sweeps_S41_S49/S41_breathing";
    // -------------------------------------------------------------------------
    const double print_dt = 100.0e-12;           // progress-line interval
    const int n_free = static_cast<int>(std::ceil(free_time / dt));
    const int sample_every = static_cast<int>(std::ceil(sample_dt / dt));
    const int snapshot_every = static_cast<int>(std::ceil(snapshot_dt / dt));
    const int print_every = static_cast<int>(std::ceil(print_dt / dt));

    for (int idx : resolve_indices(static_cast<int>(D_values.size()))) {
        const double D = D_values[idx];
        Params p = make_default_params();
        p.D = D; p.nx = nx; p.ny = ny; p.dt = dt;
        precompute(p);

        // Animation buffer created before relaxation so the relaxed
        // (pre-perturbation) configuration can be saved as a phase-0
        // frame ahead of the free-evolution drive.
        std::unique_ptr<SnapshotBuffer> snaps;
        if (dump_snapshots) {
            snaps.reset(new SnapshotBuffer(p.ny, p.nx, 500));
        }

        // Relax (local-K_eff, over-damped quench).
        SAFPair ic = saf_skyrmion(p.nx, p.ny, p.a, p.skyrmion_R, p.skyrmion_dw);
        RelaxResult rr = relax(std::move(ic.m_top), std::move(ic.m_bot), p,
                               /*demag=*/nullptr, relax_max_steps, relax_alpha,
                               relax_tol_torque, relax_tol_dE,
                               relax_check_every, print_every);

        // Perturb.
        Field3 m_top = std::move(rr.m_top);
        Field3 m_bot = std::move(rr.m_bot);
        // Relaxed equilibrium (phase 0), saved before the perturbation.
        if (snaps) {
            append_snapshot(*snaps, m_top, m_bot,
                            static_cast<int64_t>(rr.n_steps),
                            rr.n_steps * p.dt, 0, p);
        }
        radial_perturb(m_top, perturb_eps);
        radial_perturb(m_bot, perturb_eps);

        // Free LLG (paper alpha, J=0). run_trace's "drive" with a zero
        // pulse is the free evolution.
        p.alpha = free_alpha;
        p.gamma_p = p.gamma_ / (1.0 + p.alpha * p.alpha);
        RK4LocalKeffStepper stepper(p);

        RunTraceArgs ta;
        ta.p = &p;
        ta.pulse = std::make_shared<ConstantPulse>(0.0);
        ta.n_relax = 0;
        ta.n_drive = n_free;
        ta.sample_every = sample_every;
        ta.step_drive = &stepper;
        ta.step_relax = &stepper;
        ta.m_top_init = m_top;
        ta.m_bot_init = m_bot;
        ta.record_snapshot_at = -1.0;
        ta.print_every = print_every;
        ta.snapshots = snaps.get();
        ta.snapshot_every_drive = snapshot_every;
        Trace trace = run_trace(ta);

        char fn[160];
        std::snprintf(fn, sizeof(fn), "%s/breathing_keff_T%.1fns_%s.npz",
                      out_dir.c_str(), free_time * 1e9, d_tag(D).c_str());

        Metadata meta;
        meta.add("figure", std::string("breathing"));
        meta.add("demag", std::string("local_keff"));
        meta.add("field_kind", std::string("keff"));
        meta.add("D", D);
        meta.add("perturb_eps", perturb_eps);
        meta.add("free_alpha", free_alpha);
        meta.add("relax_alpha", relax_alpha);
        meta.add("relax_converged", rr.converged);
        meta.add("relax_n_steps", static_cast<long long>(rr.n_steps));
        meta.add("free_time", free_time);
        meta.add("n_free", n_free);
        meta.add("nx", nx);
        meta.add("ny", ny);
        meta.add("dt", dt);
        meta.add("sample_every", sample_every);
        save_trace(fn, trace, meta);

        if (dump_snapshots) {
            Field3 pos_top = lattice_positions(p.nx, p.ny, p.a);
            Field3 pos_bot = lattice_positions(p.nx, p.ny, p.a);
            for (int i = 0; i < p.ny; ++i)
                for (int j = 0; j < p.nx; ++j) pos_bot(i, j, 2) = -p.t_Co;
            snaps->write(snapshot_path_of(fn), pos_top, pos_bot,
                         meta.to_json());
        }
        std::printf("breathing: D=%.3f mJ/m^2 -> %s\n", D * 1e3, fn);
    }
    return 0;
}
