// Entry point. Per project convention (no argparse): all run-time
// configuration sits as named values at the top of main(); edit and
// rebuild to change a run. For sweeps, write a parallel main_*.cpp
// that sets the relevant Params fields and calls run(p).
#include "skyrmion/parameters.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/simulation.hpp"

int main() {
    using namespace skyrmion;

    Params p = make_default_params();

    // ----- Run configuration --------------------------------------------------
    p.nx = 256;
    p.ny = 256;
    p.a  = 2.0e-9;

    p.dt       = 5.0e-14;
    p.n_relax  = 10000;
    p.n_steps  = 5000;

    p.pulse = std::make_shared<ConstantPulse>(4.0e11);

    p.demag_kind = DemagKind::None;   // local-K_eff path (matches Python default)
    // For Newell demag:  p.demag_kind = DemagKind::Newell;
    //                    p.demag_accuracy = 8.0;
    //                    p.demag_tol_conv = 2.0e-2;

    p.dump_snapshots    = true;
    p.dump_every_relax  = 200;
    p.dump_every_drive  = 100;
    p.max_dump_frames   = 500;
    p.output_dir        = "output";
    p.snapshot_file     = "snapshots.npz";
    // -------------------------------------------------------------------------

    precompute(p);
    run(p);
    return 0;
}
