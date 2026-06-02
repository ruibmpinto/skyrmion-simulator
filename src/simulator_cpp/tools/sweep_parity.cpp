// Emit a C++ sweep trace at a fixed small-lattice K_eff config so the
// Python run_one driver can be compared against it element-wise.
#include "skyrmion/initial_conditions.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/sweep/driver.hpp"
#include "skyrmion/sweep/stepper.hpp"
#include "skyrmion/sweep/trace_io.hpp"

#include <cmath>
#include <memory>
#include <utility>

using namespace skyrmion;
using namespace skyrmion::sweep;

int main() {
    Params p = make_default_params();
    p.nx = 64; p.ny = 64; p.a = 2.0e-9; p.dt = 5.0e-14;
    p.D = 0.85e-3;
    p.skyrmion_R = 25.0e-9; p.skyrmion_dw = 10.0e-9;
    precompute(p);

    const double J0 = 1.0e11;
    const double t_pulse = 1.0e-10;
    const int n_relax = static_cast<int>(std::ceil(5.0e-12 / p.dt));
    const int n_drive = static_cast<int>(std::ceil(1.5e-10 / p.dt));
    const int sample_every = static_cast<int>(std::ceil(5.0e-12 / p.dt));

    p.pulse = std::make_shared<SquarePulse>(J0, 0.0, t_pulse);
    RK4LocalKeffStepper stepper(p);

    SAFPair ic = saf_skyrmion(p.nx, p.ny, p.a, p.skyrmion_R, p.skyrmion_dw);

    RunTraceArgs ta;
    ta.p = &p;
    ta.pulse = p.pulse;
    ta.n_relax = n_relax;
    ta.n_drive = n_drive;
    ta.sample_every = sample_every;
    ta.step_drive = &stepper;
    ta.step_relax = &stepper;
    ta.m_top_init = std::move(ic.m_top);
    ta.m_bot_init = std::move(ic.m_bot);
    ta.record_snapshot_at = -1.0;
    ta.print_every = 0;
    ta.snapshots = nullptr;

    Trace trace = run_trace(ta);

    Metadata meta;
    meta.add("config", std::string("sweep_parity"));
    meta.add("J0", J0);
    meta.add("n_relax", n_relax);
    meta.add("n_drive", n_drive);
    meta.add("sample_every", sample_every);
    save_trace("/tmp/cpp_sweep_parity.npz", trace, meta);
    return 0;
}
