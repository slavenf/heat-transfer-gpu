//
// Copyright (c) 2026 Slaven Falandys
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
//

#ifndef FILE_GPUSOLVEROPENCL_HPP_INCLUDED
#define FILE_GPUSOLVEROPENCL_HPP_INCLUDED

#include <memory>

#include "Real.hpp"
#include "Solver.hpp"

class Mesh;
class SolverParameters;

class GpuSolverOpenCl : public Solver
{
public:

    GpuSolverOpenCl(const Mesh& mesh, const SolverParameters& parameters);

    ~GpuSolverOpenCl() override;

    void reset() override;

    void step(std::size_t num_iterations) override;

    void draw(sf::RenderTarget& target) override;

    void draw_velocity_field(sf::RenderTarget& target) override;

    Real average_temperature() const override;

    Real max_displacement() const override;

private:

    struct Impl;

    std::unique_ptr<Impl> impl_;
};

#endif // FILE_GPUSOLVEROPENCL_HPP_INCLUDED
