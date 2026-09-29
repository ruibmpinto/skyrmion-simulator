cd "$(dirname "${BASH_SOURCE[0]}")/../src/simulator_cpp/"

module load stack/.2024-06-silent gcc/12.2.0

module load cmake

module load fftw/3.3.9

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

cmake --build build -j --target relax_track_width scan_track_width