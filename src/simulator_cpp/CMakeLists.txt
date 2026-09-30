cmake_minimum_required(VERSION 3.20)
project(skyrmion_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
             Debug Release RelWithDebInfo)

# -------------------- Build options -------------------------------------------

# Warnings are errors by default: the build is warning-clean and must stay
# so. Turn this off only to get past a new compiler's spurious diagnostic.
option(SKYRMION_WERROR "Treat compiler warnings as errors" ON)

# AddressSanitizer + UndefinedBehaviorSanitizer. Off by default because
# they cost roughly 2x runtime; intended for a Debug build:
#   cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug \
#         -DSKYRMION_SANITIZE=ON -DSKYRMION_NATIVE_ARCH=OFF
option(SKYRMION_SANITIZE "Build with address and UB sanitizers" OFF)

# -march=native pins the binary to the build machine. Required for the
# production runs; turn it off for portable or sanitizer builds.
option(SKYRMION_NATIVE_ARCH "Optimize for the build machine's CPU" ON)

# Applies the project-wide compile options to one target. Every target
# gets the same set, so they are defined in exactly one place.
function(skyrmion_target_options target)
    target_compile_options(${target} PRIVATE
        -O3 -fno-math-errno -Wall -Wextra -Wpedantic)
    if(SKYRMION_NATIVE_ARCH)
        target_compile_options(${target} PRIVATE -march=native)
    endif()
    if(SKYRMION_WERROR)
        target_compile_options(${target} PRIVATE -Werror)
    endif()
    if(SKYRMION_SANITIZE)
        target_compile_options(${target} PRIVATE
            -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE
            -fsanitize=address,undefined)
    endif()
endfunction()

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

# Minimum supported FFTW3. The version is only verified here; the include
# path and library above remain what is actually used, so this check never
# changes which FFTW is linked.
set(SKYRMION_FFTW3_MIN_VERSION 3.3.8)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PC_FFTW3 QUIET fftw3)
endif()
if(PC_FFTW3_VERSION)
    if(PC_FFTW3_VERSION VERSION_LESS SKYRMION_FFTW3_MIN_VERSION)
        message(FATAL_ERROR
                "FFTW3 ${PC_FFTW3_VERSION} is too old; "
                "${SKYRMION_FFTW3_MIN_VERSION} or newer is required.")
    endif()
    message(STATUS "FFTW3 ${PC_FFTW3_VERSION} "
                   "(minimum ${SKYRMION_FFTW3_MIN_VERSION})")
else()
    message(STATUS "FFTW3 version not reported by pkg-config; "
                   "assuming >= ${SKYRMION_FFTW3_MIN_VERSION}")
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
    src/stochastic/dissipation_partition.cpp
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

skyrmion_target_options(skyrmion_core)
target_compile_options(skyrmion_core PRIVATE -Wno-unused-parameter)

# -------------------- Executable ----------------------------------------------

add_executable(skyrmion_sim src/main.cpp)
target_link_libraries(skyrmion_sim PRIVATE skyrmion_core)
skyrmion_target_options(skyrmion_sim)

# Tool: dump Newell kernel to .npz for cross-validation against Python.
add_executable(dump_newell_kernel tools/dump_newell_kernel.cpp)
target_link_libraries(dump_newell_kernel PRIVATE skyrmion_core)
skyrmion_target_options(dump_newell_kernel)

add_executable(diff_slab_field tools/diff_slab_field.cpp)
target_link_libraries(diff_slab_field PRIVATE skyrmion_core)
skyrmion_target_options(diff_slab_field)

add_executable(sweep_parity tools/sweep_parity.cpp)
target_link_libraries(sweep_parity PRIVATE skyrmion_core)
skyrmion_target_options(sweep_parity)

add_executable(profile_run tools/profile_run.cpp)
target_link_libraries(profile_run PRIVATE skyrmion_core)
skyrmion_target_options(profile_run)

add_executable(tw_check tools/tw_check.cpp)
target_link_libraries(tw_check PRIVATE skyrmion_core)
skyrmion_target_options(tw_check)

add_executable(dump_freebc_demag tools/dump_freebc_demag.cpp)
target_link_libraries(dump_freebc_demag PRIVATE skyrmion_core)
skyrmion_target_options(dump_freebc_demag)

add_executable(bench_demag_method tools/bench_demag_method.cpp)
target_link_libraries(bench_demag_method PRIVATE skyrmion_core)
skyrmion_target_options(bench_demag_method)

add_executable(sweep_dmi_racetrack tools/sweep_dmi_racetrack.cpp)
target_link_libraries(sweep_dmi_racetrack PRIVATE skyrmion_core)
skyrmion_target_options(sweep_dmi_racetrack)

# -------------------- Sweep analyses ------------------------------------------

# One binary per source in src/sweeps/ (study-specific, kept locally).
file(GLOB SWEEP_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/src/sweeps/*.cpp)
foreach(src ${SWEEP_SOURCES})
    get_filename_component(s ${src} NAME_WE)
    add_executable(${s} ${src})
    target_link_libraries(${s} PRIVATE skyrmion_core)
    skyrmion_target_options(${s})
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
    sweep_pulse_shape
    pair_potential)
foreach(s ${STOCHASTIC_BINARIES})
    add_executable(${s} src/stochastic_runs/${s}.cpp)
    target_link_libraries(${s} PRIVATE skyrmion_core)
    skyrmion_target_options(${s})
endforeach()

# -------------------- Stochastic-LLGS validation gates ------------------------

set(VALIDATION_BINARIES
    run_equipartition
    run_t0_nodemag
    run_t0_demag
    run_langevin
    run_brown_reversal)
# Study-specific diagnostic, kept locally; built only when present.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/validation/validate_relax_torque.cpp)
    list(APPEND VALIDATION_BINARIES validate_relax_torque)
endif()
foreach(s ${VALIDATION_BINARIES})
    add_executable(${s} src/validation/${s}.cpp)
    target_link_libraries(${s} PRIVATE skyrmion_core)
    skyrmion_target_options(${s})
endforeach()

# -------------------- Parity tests --------------------------------------------

enable_testing()
set(TEST_REF_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tests/reference/references.npz")
set(SKYRMION_TESTS
    pulses parameters lattice initial_conditions
    demag_slab demag_newell demag_field
    fields energy integrator observables io_npz simulation
    sweep relax edge_cases thermal stochastic_t0 lcc stochastic_diag pair_ic
    stochastic_scans validation dissipation_partition)
foreach(t ${SKYRMION_TESTS})
    add_executable(test_${t} tests/test_${t}.cpp)
    target_include_directories(test_${t} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/include)
    target_link_libraries(test_${t} PRIVATE skyrmion_core)
    target_compile_definitions(test_${t} PRIVATE
        REFERENCE_NPZ="${TEST_REF_PATH}")
    skyrmion_target_options(test_${t})
    add_test(NAME test_${t} COMMAND test_${t})
endforeach()

# -------------------- Install -------------------------------------------------

include(GNUInstallDirs)

install(TARGETS skyrmion_core skyrmion_sim
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/skyrmion
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp")

# The sweep, production-run and validation binaries are entry points
# whose run configuration is compiled in, so they install alongside the
# main simulator rather than being treated as build-tree-only tools.
install(TARGETS ${SWEEP_BINARIES} ${STOCHASTIC_BINARIES}
                ${VALIDATION_BINARIES}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
