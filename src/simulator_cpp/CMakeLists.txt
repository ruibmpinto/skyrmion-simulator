cmake_minimum_required(VERSION 3.20)
project(skyrmion_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# -------------------- Dependencies --------------------------------------------

# libnpy (vendored at src/aux/libnpy). Path relative to this CMakeLists.txt:
#   src/simulator_cpp/CMakeLists.txt  ->  ../aux/libnpy
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../aux/libnpy
                 ${CMAKE_BINARY_DIR}/libnpy_build EXCLUDE_FROM_ALL)

# FFTW3 (system). Homebrew on Apple Silicon installs into /opt/homebrew.
find_path(FFTW3_INCLUDE_DIR fftw3.h
          HINTS /opt/homebrew/include /opt/local/include /usr/local/include)
find_library(FFTW3_LIBRARY NAMES fftw3
             HINTS /opt/homebrew/lib /opt/local/lib /usr/local/lib)
find_library(FFTW3_THREADS_LIBRARY NAMES fftw3_threads
             HINTS /opt/homebrew/lib /opt/local/lib /usr/local/lib)
if(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIBRARY)
    message(FATAL_ERROR "FFTW3 not found. brew install fftw, or set FFTW3_INCLUDE_DIR/FFTW3_LIBRARY.")
endif()

find_package(OpenMP)
find_package(Threads REQUIRED)

# -------------------- Core library --------------------------------------------

set(SKYRMION_SOURCES
    src/pulses.cpp
    src/parameters.cpp
    src/lattice.cpp
    src/initial_conditions.cpp
    src/fft2d.cpp
    src/demag_slab.cpp
    src/demag_newell.cpp
    src/demag.cpp
    src/fields.cpp
    src/energy.cpp
    src/relax_torque_probe.cpp
    src/integrator.cpp
    src/observables.cpp
    src/io_npz.cpp
    src/simulation.cpp
    src/sweep/observations.cpp
    src/sweep/relax.cpp
    src/sweep/driver.cpp
    src/sweep/trace_io.cpp
    src/stochastic/thermal.cpp
    src/stochastic/heun.cpp
    src/stochastic/lcc.cpp
    src/stochastic/diagnostics.cpp
    src/stochastic/equilibrate.cpp
    src/stochastic/trajectory.cpp
    src/stochastic/trajectory_io.cpp
    src/stochastic/pair_ic.cpp
    src/validation/analytic.cpp
    src/validation/macrospin.cpp
)

add_library(skyrmion_core STATIC ${SKYRMION_SOURCES})
# FFTW3 headers are exposed via demag.hpp -> fft2d.hpp, so consumers of
# skyrmion_core (e.g. tools/, sweep entry points) need its include path.
target_include_directories(skyrmion_core
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${FFTW3_INCLUDE_DIR})
target_link_libraries(skyrmion_core
    PUBLIC  npy::npy Threads::Threads
    PRIVATE ${FFTW3_LIBRARY})
if(FFTW3_THREADS_LIBRARY)
    target_link_libraries(skyrmion_core PRIVATE ${FFTW3_THREADS_LIBRARY})
    target_compile_definitions(skyrmion_core PRIVATE SKYRMION_FFTW_THREADS=1)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(skyrmion_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(skyrmion_core PUBLIC SKYRMION_OPENMP=1)
endif()

target_compile_options(skyrmion_core PRIVATE
    -O3 -march=native -fno-math-errno
    -Wall -Wextra -Wpedantic -Wno-unused-parameter)

# -------------------- Executable ----------------------------------------------

add_executable(skyrmion_sim src/main.cpp)
target_link_libraries(skyrmion_sim PRIVATE skyrmion_core)
target_compile_options(skyrmion_sim PRIVATE
    -O3 -march=native -fno-math-errno -Wall -Wextra -Wpedantic)

# Tool: dump Newell kernel to .npz for cross-validation against Python.
add_executable(dump_newell_kernel tools/dump_newell_kernel.cpp)
target_link_libraries(dump_newell_kernel PRIVATE skyrmion_core)
target_compile_options(dump_newell_kernel PRIVATE
    -O3 -march=native -fno-math-errno -Wall -Wextra -Wpedantic)

add_executable(diff_slab_field tools/diff_slab_field.cpp)
target_link_libraries(diff_slab_field PRIVATE skyrmion_core)
target_compile_options(diff_slab_field PRIVATE
    -O3 -march=native -fno-math-errno -Wall -Wextra -Wpedantic)

add_executable(sweep_parity tools/sweep_parity.cpp)
target_link_libraries(sweep_parity PRIVATE skyrmion_core)
target_compile_options(sweep_parity PRIVATE
    -O3 -march=native -fno-math-errno -Wall -Wextra -Wpedantic)

add_executable(profile_run tools/profile_run.cpp)
target_link_libraries(profile_run PRIVATE skyrmion_core)
target_compile_options(profile_run PRIVATE
    -O3 -march=native -fno-math-errno -Wall -Wextra -Wpedantic)

add_executable(tw_check tools/tw_check.cpp)
target_link_libraries(tw_check PRIVATE skyrmion_core)
target_compile_options(tw_check PRIVATE
    -O3 -march=native -fno-math-errno -Wall -Wextra -Wpedantic)

add_executable(dump_freebc_demag tools/dump_freebc_demag.cpp)
target_link_libraries(dump_freebc_demag PRIVATE skyrmion_core)
target_compile_options(dump_freebc_demag PRIVATE
    -O3 -march=native -fno-math-errno -Wall -Wextra -Wpedantic)

add_executable(bench_demag_method tools/bench_demag_method.cpp)
target_link_libraries(bench_demag_method PRIVATE skyrmion_core)
target_compile_options(bench_demag_method PRIVATE
    -O3 -march=native -fno-math-errno -Wall -Wextra -Wpedantic)

add_executable(sweep_dmi_racetrack tools/sweep_dmi_racetrack.cpp)
target_link_libraries(sweep_dmi_racetrack PRIVATE skyrmion_core)
target_compile_options(sweep_dmi_racetrack PRIVATE
    -O3 -march=native -fno-math-errno -Wall -Wextra -Wpedantic)

# -------------------- Sweep analyses ------------------------------------------

set(SWEEP_BINARIES
    sweep_S41_v_time
    sweep_D_S41
    sweep_D_S41_local
    sweep_S42_J
    sweep_S43a_FWHM
    sweep_S43b_J_two_D
    sweep_S44_deformation
    sweep_S46a_HRKKY
    sweep_S46b_FWHM
    sweep_S47_J_config
    sweep_S48_inertia
    sweep_S49_TSH
    sweep_breathing)
foreach(s ${SWEEP_BINARIES})
    add_executable(${s} src/sweeps/${s}.cpp)
    target_link_libraries(${s} PRIVATE skyrmion_core)
    target_compile_options(${s} PRIVATE
        -O3 -march=native -fno-math-errno -Wall -Wextra -Wpedantic)
endforeach()

# -------------------- Stochastic-LLGS production runs -------------------------

set(STOCHASTIC_BINARIES
    run_single
    scan_tj
    scan_track_width
    relax_track_width
    equilibrate_track_width
    scan_arrhenius
    scan_radius
    pair_potential)
foreach(s ${STOCHASTIC_BINARIES})
    add_executable(${s} src/stochastic_runs/${s}.cpp)
    target_link_libraries(${s} PRIVATE skyrmion_core)
    target_compile_options(${s} PRIVATE
        -O3 -march=native -fno-math-errno -Wall -Wextra -Wpedantic)
endforeach()

# -------------------- Stochastic-LLGS validation gates ------------------------

set(VALIDATION_BINARIES
    run_equipartition
    run_t0_nodemag
    run_t0_demag
    run_langevin
    run_brown_reversal
    validate_relax_torque)
foreach(s ${VALIDATION_BINARIES})
    add_executable(${s} src/validation/${s}.cpp)
    target_link_libraries(${s} PRIVATE skyrmion_core)
    target_compile_options(${s} PRIVATE
        -O3 -march=native -fno-math-errno -Wall -Wextra -Wpedantic)
endforeach()

# -------------------- Parity tests --------------------------------------------

enable_testing()
set(TEST_REF_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tests/reference/references.npz")
set(SKYRMION_TESTS
    pulses parameters lattice initial_conditions
    demag_slab demag_newell demag_field
    fields energy integrator observables io_npz simulation
    sweep relax edge_cases thermal stochastic_t0 lcc stochastic_diag pair_ic
    stochastic_scans validation)
foreach(t ${SKYRMION_TESTS})
    add_executable(test_${t} tests/test_${t}.cpp)
    target_include_directories(test_${t} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/include)
    target_link_libraries(test_${t} PRIVATE skyrmion_core)
    target_compile_definitions(test_${t} PRIVATE
        REFERENCE_NPZ="${TEST_REF_PATH}")
    target_compile_options(test_${t} PRIVATE
        -O3 -march=native -fno-math-errno -Wall -Wextra -Wpedantic)
    add_test(NAME test_${t} COMMAND test_${t})
endforeach()
