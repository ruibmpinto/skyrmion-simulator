/// \file
/// Per-trajectory trace: fine-cadence observable arrays plus an
/// optional single spin snapshot. Mirrors the dict returned by
/// src/sweeps/driver.py::run_one.
#pragma once

#include "skyrmion/sweep/observations.hpp"
#include "skyrmion/types.hpp"

#include <vector>

namespace skyrmion {
namespace sweep {

/// One trajectory's observable history.
///
/// Every vector is sampled at the same cadence and stays index-aligned
/// with t, so the arrays serialize directly into the trace .npz.
struct Trace {
    std::vector<double> t;  ///< Drive-phase sample times (s)
    std::vector<double> cx_top, cy_top, cx_bot, cy_bot;
    std::vector<double> d_top, d_bot;
    std::vector<double> D1_top, D2_top, theta_top;
    std::vector<double> D1_bot, D2_bot, theta_bot;
    std::vector<double> psi_top, psi_bot;
    std::vector<double> Q_top, Q_bot;

    /// Optional single snapshot (record_snapshot_at). Written into the
    /// trace .npz when present; the field-snapshot animation stream is
    /// a separate SnapshotBuffer handled by the driver.
    bool   has_snapshot = false;
    Field3 snapshot_m_top, snapshot_m_bot;  ///< Valid if has_snapshot
    double snapshot_t = 0.0;                ///< Snapshot time (s)

    /// Append one measured frame to every observable array.
    /// \param t_s Sample time, in seconds.
    /// \param o Observables measured at that time.
    void push(double t_s, const Observations& o) {
        t.push_back(t_s);
        cx_top.push_back(o.cx_top); cy_top.push_back(o.cy_top);
        cx_bot.push_back(o.cx_bot); cy_bot.push_back(o.cy_bot);
        d_top.push_back(o.d_top);   d_bot.push_back(o.d_bot);
        D1_top.push_back(o.D1_top); D2_top.push_back(o.D2_top);
        theta_top.push_back(o.theta_top);
        D1_bot.push_back(o.D1_bot); D2_bot.push_back(o.D2_bot);
        theta_bot.push_back(o.theta_bot);
        psi_top.push_back(o.psi_top); psi_bot.push_back(o.psi_bot);
        Q_top.push_back(o.Q_top);   Q_bot.push_back(o.Q_bot);
    }
};

} // namespace sweep
} // namespace skyrmion
