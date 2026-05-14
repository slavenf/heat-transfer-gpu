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

#ifndef FILE_CPUSOLVER_HPP_INCLUDED
#define FILE_CPUSOLVER_HPP_INCLUDED

#include <cstdint>
#include <glm/vec2.hpp>
#include <vector>

#include "Mesh.hpp"
#include "Solver.hpp"
#include "SolverParameters.hpp"

class CpuSolver : public Solver
{
public:

    CpuSolver(const Mesh& mesh, const SolverParameters& parameters);

    void reset() override;

    void step(std::size_t num_iterations) override;

    void draw(sf::RenderTarget& target) override;

    void draw_velocity_field(sf::RenderTarget& target) override;

    float average_temperature() const override;

    float max_displacement() const override;

private:

    void add_buoyancy();

    void add_heat();

    void advect_temperature();

    void advect_velocity();

    void apply_temperature_boundaries();

    void apply_velocity_boundaries();

    void diffuse_temperature();

    void diffuse_velocity();

    void project_velocity();

    float sample_temperature(float x, float y, float fallback_temperature) const;

    glm::vec2 sample_velocity(float x, float y) const;

private:

    const Mesh& mesh_;
    const SolverParameters& parameters_;

    std::vector<float> curr_temperature_;
    std::vector<float> next_temperature_;

    std::vector<glm::vec2> curr_velocity_;
    std::vector<glm::vec2> next_velocity_;

    std::vector<float> curr_pressure_;
    std::vector<float> next_pressure_;

    std::vector<float> divergence_;

    std::vector<std::uint8_t> pixels_;

    sf::Texture texture_;
};

#endif // FILE_CPUSOLVER_HPP_INCLUDED
