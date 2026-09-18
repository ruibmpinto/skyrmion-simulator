/// \file
/// Field types and primitive aliases for the skyrmion simulator.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace skyrmion {

/// Scalar floating-point type used throughout the simulator.
using Real = double;
/// Three-component Cartesian vector (x, y, z).
using Vec3 = std::array<Real, 3>;

/// Row-major (ny, nx, 3) lattice vector field. Storage matches numpy's
/// default layout, so .npz dumps load directly into a numpy array of
/// shape (ny, nx, 3).
struct Field3 {
    int ny = 0;  ///< Number of lattice rows (y direction).
    int nx = 0;  ///< Number of lattice columns (x direction).
    std::vector<Real> data;  ///< ny * nx * 3 values, row-major.

    /// Empty field: no rows, no columns, no storage.
    Field3() = default;
    /// Zero-filled field of shape (ny_, nx_, 3).
    /// \param ny_ Number of lattice rows.
    /// \param nx_ Number of lattice columns.
    Field3(int ny_, int nx_)
        : ny(ny_), nx(nx_),
          data(static_cast<std::size_t>(ny_) * static_cast<std::size_t>(nx_) * 3, 0.0) {}

    /// Mutable component `k` of the vector at row `i`, column `j`.
    inline Real& operator()(int i, int j, int k) {
        return data[(static_cast<std::size_t>(i) * nx + j) * 3 + k];
    }
    /// Component `k` of the vector at row `i`, column `j`.
    inline Real operator()(int i, int j, int k) const {
        return data[(static_cast<std::size_t>(i) * nx + j) * 3 + k];
    }

    /// Pointer to the three contiguous components at (i, j).
    inline Real*       site_ptr(int i, int j)       { return data.data() + ((static_cast<std::size_t>(i) * nx + j) * 3); }
    /// Read-only pointer to the three components at (i, j).
    inline const Real* site_ptr(int i, int j) const { return data.data() + ((static_cast<std::size_t>(i) * nx + j) * 3); }

    /// \return Number of lattice sites, ny * nx.
    std::size_t n_sites() const { return static_cast<std::size_t>(ny) * nx; }
};

/// Row-major (ny, nx) scalar field. Used internally by demag kernels.
struct Field1 {
    int ny = 0;  ///< Number of lattice rows (y direction).
    int nx = 0;  ///< Number of lattice columns (x direction).
    std::vector<Real> data;  ///< ny * nx values, row-major.

    /// Empty field: no rows, no columns, no storage.
    Field1() = default;
    /// Zero-filled field of shape (ny_, nx_).
    /// \param ny_ Number of lattice rows.
    /// \param nx_ Number of lattice columns.
    Field1(int ny_, int nx_)
        : ny(ny_), nx(nx_),
          data(static_cast<std::size_t>(ny_) * static_cast<std::size_t>(nx_), 0.0) {}

    /// Mutable value at row `i`, column `j`.
    inline Real& operator()(int i, int j)       { return data[static_cast<std::size_t>(i) * nx + j]; }
    /// Value at row `i`, column `j`.
    inline Real  operator()(int i, int j) const { return data[static_cast<std::size_t>(i) * nx + j]; }
};

/// Inline vector helpers (kept short — heavy use in hot loops).
///
/// Cross product out = a x b. `out` must not alias `a` or `b`.
/// \param a Left operand, three components.
/// \param b Right operand, three components.
/// \param out Destination, three components.
inline void cross3(const Real a[3], const Real b[3], Real out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

/// \return Dot product a . b.
inline Real dot3(const Real a[3], const Real b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/// \return Euclidean norm |a|.
inline Real norm3(const Real a[3]) {
    return std::sqrt(dot3(a, a));
}

} // namespace skyrmion
