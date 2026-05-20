# GPU-Accelerated Heat Transfer Simulator

This is a 2D heat transfer simulator implemented in C++ using GPU acceleration. It simulates heat conduction in fluids and solids, and heat convection in fluids. It supports NVIDIA, AMD, and Intel GPUs.

The project provides a CPU solver and two GPU-based solvers:
- **OpenCL** solver
- **OpenGL** solver based on compute shaders

The simulation models heat transfer on a rectangular grid using a finite-difference method.

![](doc/heat-transfer.gif)

![](doc/water.gif)

## Build Instructions

The project has been tested on Linux.

Required build tools:
- GCC
- GNU Make
- xxd

Required libraries:
- SFML 3
- OpenMP
- OpenCL (OpenCL header files and OpenCL ICD loader)
- OpenGL

To build the project, run the `make` command from the project's root directory:
```
make
```

## Usage

Run the simulator:
```
./build/heat-transfer cases/metal-plate-400x200.json
```

By default, the CPU solver is used.

To use the GPU solver based on OpenCL, run:
```
./build/heat-transfer cases/metal-plate-400x200.json -s opencl
```

To use the GPU solver based on OpenGL, run:
```
./build/heat-transfer cases/metal-plate-400x200.json -s opengl
```

Another example case:
```
./build/heat-transfer cases/water-150x150.json
```

If the rendering window is too small, use `-z` to increase its size, for example:
```
./build/heat-transfer cases/water-150x150.json -z 4
```

For more information, run:
```
./build/heat-transfer --help
```

Controls:
- `Esc` - close the window
- `R` - reset the simulation
- `H` - toggle the HUD
- `V` - toggle the velocity field

## Double Precision

By default, the simulator is built with single-precision `float` values. To build with double-precision `double` values, set `USE_DOUBLE=1`:
```
make clean
make USE_DOUBLE=1
```

Double precision is supported by the CPU and OpenCL solvers. The OpenCL device must support `cl_khr_fp64`. The OpenGL solver does not support double-precision.
