// Round-trip test for the SnapshotBuffer + .npz writer. No Python
// counterpart exists (the Python code emits LAMMPS dumps); this test
// instead verifies that what the buffer writes can be read back with
// the exact values it received.
#include "skyrmion/io_npz.hpp"
#include "test_common.hpp"

#include <npy/npy.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace skyrmion;
using test_common::TestRef;
using test_common::TestRunner;
using test_common::scalar;

int main() {
    TestRef ref(REFERENCE_NPZ);
    TestRunner r;
    const int ny = static_cast<int>(ref.scalar<int64_t>("state_ny"));
    const int nx = static_cast<int>(ref.scalar<int64_t>("state_nx"));
    Field3 m_top = ref.field3("state_m_top");
    Field3 m_bot = ref.field3("state_m_bot");
    Field3 pos_top = ref.field3("state_pos_top");
    Field3 pos_bot = ref.field3("state_pos_bot");

    SnapshotBuffer buf(ny, nx, 4);
    buf.append(m_top, m_bot, 0, 0.0, 0, 0.1, -0.2, 1e-9, 2e-9, 5e-8, 7e-8, 6e-8, 0.05, 0.1);
    buf.append(m_top, m_bot, 100, 5e-12, 0, 0.9, -0.8, 1.2e-9, 2.2e-9, 5.5e-8, 7.5e-8, 6.5e-8, 0.06, 0.2);
    buf.append(m_top, m_bot, 200, 1e-11, 1, 0.95, -0.85, 1.4e-9, 2.4e-9, 5.7e-8, 7.7e-8, 6.7e-8, 0.07, 0.3);

    std::filesystem::path out = std::filesystem::temp_directory_path()
                                / "skyrmion_test_io.npz";
    std::filesystem::remove(out);
    buf.write(out.string(), pos_top, pos_bot, "{\"test\":true}");

    // Read back with libnpy.
    npy::npzfilereader rdr(out.string());
    auto m_top_back = rdr.read<npy::tensor<double>>("m_top");
    auto m_bot_back = rdr.read<npy::tensor<double>>("m_bot");
    auto step_back  = rdr.read<npy::tensor<int64_t>>("step");
    auto phase_back = rdr.read<npy::tensor<int32_t>>("phase_id");
    auto Q_top_back = rdr.read<npy::tensor<double>>("Q_top");
    auto pos_top_back = rdr.read<npy::tensor<double>>("pos_top");

    // Shapes
    r.check("m_top_shape0",  test_common::scalar(static_cast<double>(m_top_back.shape()[0]), 3.0));
    r.check("m_top_shape1",  test_common::scalar(static_cast<double>(m_top_back.shape()[1]), static_cast<double>(ny)));
    r.check("m_top_shape2",  test_common::scalar(static_cast<double>(m_top_back.shape()[2]), static_cast<double>(nx)));
    r.check("m_top_shape3",  test_common::scalar(static_cast<double>(m_top_back.shape()[3]), 3.0));

    // m_top frame 0 must equal the input.
    {
        const std::size_t per = static_cast<std::size_t>(ny) * nx * 3;
        r.check("m_top_frame0",
                test_common::array(m_top_back.data(), m_top.data.data(), per));
        r.check("m_bot_frame0",
                test_common::array(m_bot_back.data(), m_bot.data.data(), per));
    }
    // step, phase_id roundtrip
    r.check("step[0]", test_common::scalar(static_cast<double>(step_back.data()[0]), 0.0));
    r.check("step[1]", test_common::scalar(static_cast<double>(step_back.data()[1]), 100.0));
    r.check("step[2]", test_common::scalar(static_cast<double>(step_back.data()[2]), 200.0));
    r.check("phase_id[0]", test_common::scalar(static_cast<double>(phase_back.data()[0]), 0.0));
    r.check("phase_id[2]", test_common::scalar(static_cast<double>(phase_back.data()[2]), 1.0));
    // observable roundtrip
    r.check("Q_top[0]", test_common::scalar(Q_top_back.data()[0], 0.1));
    r.check("Q_top[2]", test_common::scalar(Q_top_back.data()[2], 0.95));
    // pos_top roundtrip
    r.check("pos_top",
            test_common::array(pos_top_back.data(), pos_top.data.data(), pos_top.data.size()));
    std::filesystem::remove(out);
    return r.report("test_io_npz");
}
