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

#include "CpuSolver.hpp"

#include <algorithm>

#include "temperature_to_color.hpp"

CpuSolver::CpuSolver(const Mesh& mesh)
    : mesh_(mesh)
    , curr_(mesh.width() * mesh.height())
    , next_(mesh.width() * mesh.height())
    , pixels_(mesh.width() * mesh.height() * 4)
    , texture_(sf::Vector2u(mesh.width(), mesh.height()))
{
    reset();
}

void CpuSolver::reset()
{
    #pragma omp parallel for schedule(static)
    for (std::size_t y = 0; y < mesh_.height(); ++y)
    {
        for (std::size_t x = 0; x < mesh_.width(); ++x)
        {
            const std::size_t i = mesh_.index(x, y);

            switch (mesh_.type(i))
            {
                case CellType::Vacuum:
                {
                    // Vacuum has no inital temeprature
                    break;
                }

                case CellType::Source:
                {
                    curr_[i] = 100.0f;
                    break;
                }

                case CellType::Metal:
                {
                    curr_[i] = 0.0f;
                    break;
                }
            }
        }
    }
}

void CpuSolver::step(std::size_t num_iterations)
{
    while (num_iterations-- > 0)
    {
        #pragma omp parallel for schedule(static)
        for (std::size_t y = 0; y < mesh_.height(); ++y)
        {
            for (std::size_t x = 0; x < mesh_.width(); ++x)
            {
                const std::size_t i = mesh_.index(x, y);

                switch (mesh_.type(i))
                {
                    case CellType::Vacuum:
                    case CellType::Source:
                    {
                        next_[i] = curr_[i];
                        break;
                    }

                    case CellType::Metal:
                    {
                        const float center = curr_[i];

                        float sum = 0.0f;

                        if (x > 0)
                        {
                            const std::size_t j = mesh_.index(x - 1, y);
                            if (mesh_.is_conductive(j))
                            {
                                sum += curr_[j] - center;
                            }
                        }

                        if (x + 1 < mesh_.width())
                        {
                            const std::size_t j = mesh_.index(x + 1, y);
                            if (mesh_.is_conductive(j))
                            {
                                sum += curr_[j] - center;
                            }
                        }

                        if (y > 0)
                        {
                            const std::size_t j = mesh_.index(x, y - 1);
                            if (mesh_.is_conductive(j))
                            {
                                sum += curr_[j] - center;
                            }
                        }

                        if (y + 1 < mesh_.height())
                        {
                            const std::size_t j = mesh_.index(x, y + 1);
                            if (mesh_.is_conductive(j))
                            {
                                sum += curr_[j] - center;
                            }
                        }

                        next_[i] = center + diffusion_ * sum;

                        break;
                    }
                }
            }
        }

        std::swap(curr_, next_);
    }
}

void CpuSolver::draw(sf::RenderTarget& target)
{
    #pragma omp parallel for schedule(static)
    for (std::size_t y = 0; y < mesh_.height(); ++y)
    {
        for (std::size_t x = 0; x < mesh_.width(); ++x)
        {
            const std::size_t i = mesh_.index(x, y);

            sf::Color color = sf::Color::Black;

            switch (mesh_.type(i))
            {
                case CellType::Vacuum:
                {
                    color = sf::Color::Black;
                    break;
                }

                case CellType::Source:
                {
                    color = sf::Color::White;
                    break;
                }

                case CellType::Metal:
                {
                    color = temperature_to_color(curr_[i], 0.0f, 100.0f);
                    break;
                }
            }

            pixels_[i * 4 + 0] = color.r;
            pixels_[i * 4 + 1] = color.g;
            pixels_[i * 4 + 2] = color.b;
            pixels_[i * 4 + 3] = 255;
        }
    }

    texture_.update(pixels_.data());

    target.draw(sf::Sprite(texture_));
}
