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
#include <vector>

#include "Mesh.hpp"
#include "Solver.hpp"

class CpuSolver : public Solver
{
public:

    CpuSolver(const Mesh& mesh);

    void reset() override;

    void step(std::size_t num_iterations) override;

    void draw(sf::RenderTarget& target) override;

private:

    const Mesh& mesh_;
    std::vector<float> curr_;
    std::vector<float> next_;

    float diffusion_ = 0.2f;

    std::vector<std::uint8_t> pixels_;

    sf::Texture texture_;
};

#endif // FILE_CPUSOLVER_HPP_INCLUDED
