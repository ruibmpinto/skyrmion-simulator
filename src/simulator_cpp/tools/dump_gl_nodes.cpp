// Dumps Gauss-Legendre nodes/weights for cross-checking against scipy.
#include <cstdio>
#include <vector>

// Mirror the static implementation in demag_newell.cpp.
namespace {
constexpr double kPi = 3.14159265358979323846;
void compute_gl_nodes(int n, std::vector<double>& nodes,
                      std::vector<double>& weights) {
    nodes.assign(n, 0.0);
    weights.assign(n, 0.0);
    if (n == 1) { nodes[0] = 0.0; weights[0] = 1.0; return; }
    for (int i = 0; i < n; ++i) {
        double x = std::cos(kPi * (4.0 * (i + 1) - 1.0) / (4.0 * n + 2.0));
        for (int it = 0; it < 200; ++it) {
            double pm1 = 0.0, p0 = 1.0, p1 = 0.0;
            for (int k = 1; k <= n; ++k) {
                p1 = ((2.0 * k - 1.0) * x * p0 - (k - 1.0) * pm1) / k;
                pm1 = p0; p0 = p1;
            }
            double Pn_prime = n * (x * p0 - pm1) / (x * x - 1.0);
            double dx = p0 / Pn_prime;
            x -= dx;
            if (std::abs(dx) < 1e-15) break;
        }
        nodes[i] = x;
        double pm1 = 0.0, p0 = 1.0, p1 = 0.0;
        for (int k = 1; k <= n; ++k) {
            p1 = ((2.0 * k - 1.0) * x * p0 - (k - 1.0) * pm1) / k;
            pm1 = p0; p0 = p1;
        }
        double Pn_prime = n * (x * p0 - pm1) / (x * x - 1.0);
        weights[i] = 2.0 / ((1.0 - x * x) * Pn_prime * Pn_prime);
        weights[i] *= 0.5;
    }
}
} // namespace

int main() {
    for (int n : {4, 8, 16}) {
        std::vector<double> nodes, weights;
        compute_gl_nodes(n, nodes, weights);
        std::printf("n=%d\n", n);
        for (int i = 0; i < n; ++i) {
            std::printf("  x[%d]=%+.17e  w[%d]=%.17e\n", i, nodes[i], i, weights[i]);
        }
    }
    return 0;
}
