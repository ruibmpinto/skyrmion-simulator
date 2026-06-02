// Field types and primitive aliases for the skyrmion simulator.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace skyrmion {

using Real = double;
using Vec3 = std::array<Real, 3>;

// Row-major (ny, nx, 3) lattice vector field. Storage matches numpy's
// default layout, so .npz dumps load directly into a numpy array of
// shape (ny, nx, 3).
struct Field3 {
    int ny = 0;
    int nx = 0;
    std::vector<Real> data;

    Field3() = default;
    Field3(int ny_, int nx_)
        : ny(ny_), nx(nx_),
          data(static_cast<std::size_t>(ny_) * static_cast<std::size_t>(nx_) * 3, 0.0) {}

    inline Real& operator()(int i, int j, int k) {
        return data[(static_cast<std::size_t>(i) * nx + j) * 3 + k];
    }
    inline Real operator()(int i, int j, int k) const {
        return data[(static_cast<std::size_t>(i) * nx + j) * 3 + k];
    }

    inline Real*       site_ptr(int i, int j)       { return data.data() + ((static_cast<std::size_t>(i) * nx + j) * 3); }
    inline const Real* site_ptr(int i, int j) const { return data.data() + ((static_cast<std::size_t>(i) * nx + j) * 3); }

    std::size_t n_sites() const { return static_cast<std::size_t>(ny) * nx; }
};

// Row-major (ny, nx) scalar field. Used internally by demag kernels.
struct Field1 {
    int ny = 0;
    int nx = 0;
    std::vector<Real> data;

    Field1() = default;
    Field1(int ny_, int nx_)
        : ny(ny_), nx(nx_),
          data(static_cast<std::size_t>(ny_) * static_cast<std::size_t>(nx_), 0.0) {}

    inline Real& operator()(int i, int j)       { return data[static_cast<std::size_t>(i) * nx + j]; }
    inline Real  operator()(int i, int j) const { return data[static_cast<std::size_t>(i) * nx + j]; }
};

// Inline vector helpers (kept short — heavy use in hot loops).
inline void cross3(const Real a[3], const Real b[3], Real out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

inline Real dot3(const Real a[3], const Real b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline Real norm3(const Real a[3]) {
    return std::sqrt(dot3(a, a));
}

} // namespace skyrmion
