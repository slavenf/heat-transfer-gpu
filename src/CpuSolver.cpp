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
#include <glm/geometric.hpp>
#include <cmath>

#include "temperature_to_color.hpp"

CpuSolver::CpuSolver(const Mesh& mesh, const SolverParameters& parameters)
    : mesh_(mesh)
    , parameters_(parameters)
    , curr_temperature_(mesh.width() * mesh.height())
    , next_temperature_(mesh.width() * mesh.height())
    , curr_velocity_(mesh.width() * mesh.height())
    , next_velocity_(mesh.width() * mesh.height())
    , curr_pressure_(mesh.width() * mesh.height())
    , next_pressure_(mesh.width() * mesh.height())
    , divergence_(mesh.width() * mesh.height())
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
            const auto i = mesh_.index(x, y);

            curr_temperature_[i] = mesh_.initial_temperature(i);
            next_temperature_[i] = mesh_.initial_temperature(i);

            curr_velocity_[i] = glm::vec2(0.0f, 0.0f);
            next_velocity_[i] = glm::vec2(0.0f, 0.0f);

            curr_pressure_[i] = 0.0f;
            next_pressure_[i] = 0.0f;

            divergence_[i] = 0.0f;
        }
    }
}

void CpuSolver::step(std::size_t num_iterations)
{
    #pragma omp parallel firstprivate(num_iterations)
    {
        while (num_iterations-- > 0)
        {
            add_heat();

            add_buoyancy();

            apply_velocity_boundaries();

            advect_velocity();

            apply_velocity_boundaries();

            diffuse_velocity();

            apply_velocity_boundaries();

            project_velocity();

            apply_velocity_boundaries();

            advect_temperature();

            diffuse_temperature();

            apply_temperature_boundaries();
        }
    }
}

void CpuSolver::draw(sf::RenderTarget& target)
{
    #pragma omp parallel for schedule(static)
    for (std::size_t y = 0; y < mesh_.height(); ++y)
    {
        for (std::size_t x = 0; x < mesh_.width(); ++x)
        {
            sf::Color color = sf::Color::Black;

            const auto i = mesh_.index(x, y);
            const auto ti = mesh_.type(i);

            switch (ti)
            {
                case CellType::Source:
                {
                    color = sf::Color::White;
                    break;
                }

                case CellType::Vacuum:
                {
                    color = sf::Color::Black;
                    break;
                }

                case CellType::Solid:
                case CellType::Fluid:
                {
                    color = temperature_to_color
                    (
                        curr_temperature_[i],
                        mesh_.min_temperature(),
                        mesh_.max_temperature()
                    );
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

void CpuSolver::draw_velocity_field(sf::RenderTarget& target)
{
    sf::VertexArray lines(sf::PrimitiveType::Lines);

    for (std::size_t y = 0; y < mesh_.height(); y += 2)
    {
        for (std::size_t x = 0; x < mesh_.width(); x += 2)
        {
            const auto i = mesh_.index(x, y);
            const auto ti = mesh_.type(i);

            if (ti == CellType::Fluid)
            {
                const glm::vec2 start(x + 0.5f, y + 0.5f);
                const glm::vec2 end = start + normalize(curr_velocity_[i]);

                lines.append(sf::Vertex({start.x, start.y}, sf::Color::White));
                lines.append(sf::Vertex({end.x,   end.y},   sf::Color::White));
            }
        }
    }

    target.draw(lines);
}

float CpuSolver::average_temperature() const
{
    float sum = 0.0f;
    std::size_t count = 0;

    for (std::size_t i = 0; i < curr_temperature_.size(); ++i)
    {
        const auto ti = mesh_.type(i);

        if (ti == CellType::Fluid || ti == CellType::Solid)
        {
            sum += curr_temperature_[i];
            ++count;
        }
    }

    return sum / count;
}

float CpuSolver::max_displacement() const
{
    float max_speed = 0.0f;

    for (std::size_t i = 0; i < curr_velocity_.size(); ++i)
    {
        const auto ti = mesh_.type(i);

        if (ti == CellType::Fluid)
        {
            float speed = length(curr_velocity_[i]);

            if (speed > max_speed)
            {
                max_speed = speed;
            }
        }
    }

    return max_speed * parameters_.dt;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void CpuSolver::add_buoyancy()
{
    const float ambient_temperature = mesh_.min_temperature();

    #pragma omp for schedule(static)
    for (std::size_t y = 0; y < mesh_.height(); ++y)
    {
        for (std::size_t x = 0; x < mesh_.width(); ++x)
        {
            const auto i = mesh_.index(x, y);
            const auto ti = mesh_.type(i);

            switch (ti)
            {
                case CellType::Fluid:
                {
                    const auto delta_temperature = curr_temperature_[i] - ambient_temperature;

                    curr_velocity_[i].y -= parameters_.buoyancy * delta_temperature * parameters_.dt;

                    break;
                }

                default:
                    break;
            }
        }
    }
}

void CpuSolver::add_heat()
{
    #pragma omp for schedule(static)
    for (std::size_t y = 0; y < mesh_.height(); ++y)
    {
        for (std::size_t x = 0; x < mesh_.width(); ++x)
        {
            const auto i = mesh_.index(x, y);
            const auto ti = mesh_.type(i);

            switch (ti)
            {
                case CellType::Solid:
                case CellType::Fluid:
                {
                    bool touches_source = false;

                    if (x > 0)
                    {
                        const auto j = mesh_.index(x - 1, y);
                        const auto tj = mesh_.type(j);
                        if (tj == CellType::Source)
                        {
                            touches_source = true;
                        }
                    }

                    if (x + 1 < mesh_.width())
                    {
                        const auto j = mesh_.index(x + 1, y);
                        const auto tj = mesh_.type(j);
                        if (tj == CellType::Source)
                        {
                            touches_source = true;
                        }
                    }

                    if (y > 0)
                    {
                        const auto j = mesh_.index(x, y - 1);
                        const auto tj = mesh_.type(j);
                        if (tj == CellType::Source)
                        {
                            touches_source = true;
                        }
                    }

                    if (y + 1 < mesh_.height())
                    {
                        const auto j = mesh_.index(x, y + 1);
                        const auto tj = mesh_.type(j);
                        if (tj == CellType::Source)
                        {
                            touches_source = true;
                        }
                    }

                    if (touches_source)
                    {
                        curr_temperature_[i] += parameters_.source_heat_transfer * parameters_.dt;

                        if (curr_temperature_[i] > mesh_.max_temperature())
                        {
                            curr_temperature_[i] = mesh_.max_temperature();
                        }
                    }

                    break;
                }

                default:
                {
                    break;
                }
            }
        }
    }
}

void CpuSolver::advect_temperature()
{
    #pragma omp for schedule(static)
    for (std::size_t y = 0; y < mesh_.height(); ++y)
    {
        for (std::size_t x = 0; x < mesh_.width(); ++x)
        {
            const auto i = mesh_.index(x, y);
            const auto ti = mesh_.type(i);

            switch (ti)
            {
                case CellType::Fluid:
                {
                    const auto old_x = std::clamp
                    (
                        static_cast<float>(x) - curr_velocity_[i].x * parameters_.dt,
                        0.0f,
                        static_cast<float>(mesh_.width() - 1)
                    );

                    const auto old_y = std::clamp
                    (
                        static_cast<float>(y) - curr_velocity_[i].y * parameters_.dt,
                        0.0f,
                        static_cast<float>(mesh_.height() - 1)
                    );

                    next_temperature_[i] = sample_temperature(old_x, old_y, curr_temperature_[i]);

                    break;
                }

                default:
                {
                    next_temperature_[i] = curr_temperature_[i];
                    break;
                }
            }
        }
    }

    #pragma omp single
    std::swap(curr_temperature_, next_temperature_);
}

void CpuSolver::advect_velocity()
{
    #pragma omp for schedule(static)
    for (std::size_t y = 0; y < mesh_.height(); ++y)
    {
        for (std::size_t x = 0; x < mesh_.width(); ++x)
        {
            const auto i = mesh_.index(x, y);
            const auto ti = mesh_.type(i);

            switch (ti)
            {
                case CellType::Fluid:
                {
                    const auto old_x = std::clamp
                    (
                        static_cast<float>(x) - curr_velocity_[i].x * parameters_.dt,
                        0.0f,
                        static_cast<float>(mesh_.width() - 1)
                    );

                    const auto old_y = std::clamp
                    (
                        static_cast<float>(y) - curr_velocity_[i].y * parameters_.dt,
                        0.0f,
                        static_cast<float>(mesh_.height() - 1)
                    );

                    next_velocity_[i] = sample_velocity(old_x, old_y);
                    next_velocity_[i] *= parameters_.velocity_damping;

                    break;
                }

                default:
                {
                    next_velocity_[i] = glm::vec2(0.0f, 0.0f);
                    break;
                }
            }
        }
    }

    #pragma omp single
    std::swap(curr_velocity_, next_velocity_);
}

void CpuSolver::apply_temperature_boundaries()
{
    #pragma omp for schedule(static)
    for (std::size_t y = 0; y < mesh_.height(); ++y)
    {
        for (std::size_t x = 0; x < mesh_.width(); ++x)
        {
            const auto i = mesh_.index(x, y);
            const auto ti = mesh_.type(i);

            switch (ti)
            {
                case CellType::Solid:
                case CellType::Fluid:
                {
                    curr_temperature_[i] = std::clamp
                    (
                        curr_temperature_[i],
                        mesh_.min_temperature(),
                        mesh_.max_temperature()
                    );
                    break;
                }

                case CellType::Source:
                {
                    curr_temperature_[i] = mesh_.max_temperature();
                    break;
                }

                default:
                {
                    break;
                }
            }
        }
    }
}

void CpuSolver::apply_velocity_boundaries()
{
    #pragma omp for schedule(static)
    for (std::size_t y = 0; y < mesh_.height(); ++y)
    {
        for (std::size_t x = 0; x < mesh_.width(); ++x)
        {
            const auto i = mesh_.index(x, y);
            const auto ti = mesh_.type(i);

            switch (ti)
            {
                case CellType::Fluid:
                {
                    if (x > 0)
                    {
                        const auto j = mesh_.index(x - 1, y);
                        const auto tj = mesh_.type(j);
                        if (tj != CellType::Fluid)
                        {
                            if (curr_velocity_[i].x < 0.0f)
                            {
                                curr_velocity_[i].x = 0.0f;
                            }
                        }
                    }
                    else if (curr_velocity_[i].x < 0.0f)
                    {
                        curr_velocity_[i].x = 0.0f;
                    }

                    if (x + 1 < mesh_.width())
                    {
                        const auto j = mesh_.index(x + 1, y);
                        const auto tj = mesh_.type(j);
                        if (tj != CellType::Fluid)
                        {
                            if (curr_velocity_[i].x > 0.0f)
                            {
                                curr_velocity_[i].x = 0.0f;
                            }
                        }
                    }
                    else if (curr_velocity_[i].x > 0.0f)
                    {
                        curr_velocity_[i].x = 0.0f;
                    }

                    if (y > 0)
                    {
                        const auto j = mesh_.index(x, y - 1);
                        const auto tj = mesh_.type(j);
                        if (tj != CellType::Fluid)
                        {
                            if (curr_velocity_[i].y < 0.0f)
                            {
                                curr_velocity_[i].y = 0.0f;
                            }
                        }
                    }
                    else if (curr_velocity_[i].y < 0.0f)
                    {
                        curr_velocity_[i].y = 0.0f;
                    }

                    if (y + 1 < mesh_.height())
                    {
                        const auto j = mesh_.index(x, y + 1);
                        const auto tj = mesh_.type(j);
                        if (tj != CellType::Fluid)
                        {
                            if (curr_velocity_[i].y > 0.0f)
                            {
                                curr_velocity_[i].y = 0.0f;
                            }
                        }
                    }
                    else if (curr_velocity_[i].y > 0.0f)
                    {
                        curr_velocity_[i].y = 0.0f;
                    }

                    break;
                }

                default:
                {
                    curr_velocity_[i] = glm::vec2(0.0f, 0.0f);
                    break;
                }
            }
        }
    }
}

void CpuSolver::diffuse_temperature()
{
    #pragma omp for schedule(static)
    for (std::size_t y = 0; y < mesh_.height(); ++y)
    {
        for (std::size_t x = 0; x < mesh_.width(); ++x)
        {
            const auto i = mesh_.index(x, y);
            const auto ti = mesh_.type(i);

            switch (ti)
            {
                case CellType::Solid:
                case CellType::Fluid:
                {
                    float sum = 0.0f;

                    if (x > 0)
                    {
                        const auto j = mesh_.index(x - 1, y);
                        const auto tj = mesh_.type(j);
                        if (tj == CellType::Solid || tj == CellType::Fluid || tj == CellType::Source)
                        {
                            sum += curr_temperature_[j] - curr_temperature_[i];
                        }
                    }

                    if (x + 1 < mesh_.width())
                    {
                        const auto j = mesh_.index(x + 1, y);
                        const auto tj = mesh_.type(j);
                        if (tj == CellType::Solid || tj == CellType::Fluid || tj == CellType::Source)
                        {
                            sum += curr_temperature_[j] - curr_temperature_[i];
                        }
                    }

                    if (y > 0)
                    {
                        const auto j = mesh_.index(x, y - 1);
                        const auto tj = mesh_.type(j);
                        if (tj == CellType::Solid || tj == CellType::Fluid || tj == CellType::Source)
                        {
                            sum += curr_temperature_[j] - curr_temperature_[i];
                        }
                    }

                    if (y + 1 < mesh_.height())
                    {
                        const auto j = mesh_.index(x, y + 1);
                        const auto tj = mesh_.type(j);
                        if (tj == CellType::Solid || tj == CellType::Fluid || tj == CellType::Source)
                        {
                            sum += curr_temperature_[j] - curr_temperature_[i];
                        }
                    }

                    next_temperature_[i] = curr_temperature_[i] + sum * parameters_.thermal_diffusion * parameters_.dt;

                    break;
                }

                default:
                {
                    next_temperature_[i] = curr_temperature_[i];
                    break;
                }
            }
        }
    }

    #pragma omp single
    std::swap(curr_temperature_, next_temperature_);
}

void CpuSolver::diffuse_velocity()
{
    #pragma omp for schedule(static)
    for (std::size_t y = 0; y < mesh_.height(); ++y)
    {
        for (std::size_t x = 0; x < mesh_.width(); ++x)
        {
            const auto i = mesh_.index(x, y);
            const auto ti = mesh_.type(i);

            switch (ti)
            {
                case CellType::Fluid:
                {
                    glm::vec2 sum(0.0f, 0.0f);

                    if (x > 0)
                    {
                        const auto j = mesh_.index(x - 1, y);
                        const auto tj = mesh_.type(j);
                        if (tj == CellType::Fluid)
                        {
                            sum += curr_velocity_[j] - curr_velocity_[i];
                        }
                    }

                    if (x + 1 < mesh_.width())
                    {
                        const auto j = mesh_.index(x + 1, y);
                        const auto tj = mesh_.type(j);
                        if (tj == CellType::Fluid)
                        {
                            sum += curr_velocity_[j] - curr_velocity_[i];
                        }
                    }

                    if (y > 0)
                    {
                        const auto j = mesh_.index(x, y - 1);
                        const auto tj = mesh_.type(j);
                        if (tj == CellType::Fluid)
                        {
                            sum += curr_velocity_[j] - curr_velocity_[i];
                        }
                    }

                    if (y + 1 < mesh_.height())
                    {
                        const auto j = mesh_.index(x, y + 1);
                        const auto tj = mesh_.type(j);
                        if (tj == CellType::Fluid)
                        {
                            sum += curr_velocity_[j] - curr_velocity_[i];
                        }
                    }

                    next_velocity_[i] = curr_velocity_[i] + sum * parameters_.viscosity * parameters_.dt;

                    break;
                }

                default:
                {
                    next_velocity_[i] = glm::vec2(0.0f, 0.0f);
                    break;
                }
            }
        }
    }

    #pragma omp single
    std::swap(curr_velocity_, next_velocity_);
}

void CpuSolver::project_velocity()
{
    ///////////////////////////////////////////////////////////////////////////
    // Compute divergence
    ///////////////////////////////////////////////////////////////////////////

    #pragma omp for schedule(static)
    for (std::size_t y = 0; y < mesh_.height(); ++y)
    {
        for (std::size_t x = 0; x < mesh_.width(); ++x)
        {
            const auto i = mesh_.index(x, y);
            const auto ti = mesh_.type(i);

            switch (ti)
            {
                case CellType::Fluid:
                {
                    glm::vec2 left_velocity(0.0f, 0.0f);
                    glm::vec2 right_velocity(0.0f, 0.0f);
                    glm::vec2 up_velocity(0.0f, 0.0f);
                    glm::vec2 down_velocity(0.0f, 0.0f);

                    if (x > 0)
                    {
                        const auto j = mesh_.index(x - 1, y);
                        const auto tj = mesh_.type(j);
                        if (tj == CellType::Fluid)
                        {
                            left_velocity = curr_velocity_[j];
                        }
                    }

                    if (x + 1 < mesh_.width())
                    {
                        const auto j = mesh_.index(x + 1, y);
                        const auto tj = mesh_.type(j);
                        if (tj == CellType::Fluid)
                        {
                            right_velocity = curr_velocity_[j];
                        }
                    }

                    if (y > 0)
                    {
                        const auto j = mesh_.index(x, y - 1);
                        const auto tj = mesh_.type(j);
                        if (tj == CellType::Fluid)
                        {
                            up_velocity = curr_velocity_[j];
                        }
                    }

                    if (y + 1 < mesh_.height())
                    {
                        const auto j = mesh_.index(x, y + 1);
                        const auto tj = mesh_.type(j);
                        if (tj == CellType::Fluid)
                        {
                            down_velocity = curr_velocity_[j];
                        }
                    }

                    divergence_[i] =
                        0.5f *
                        (
                            right_velocity.x - left_velocity.x +
                            down_velocity.y - up_velocity.y
                        );

                    break;
                }

                default:
                {
                    divergence_[i] = 0.0f;
                    break;
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Solve pressure
    ///////////////////////////////////////////////////////////////////////////

    #pragma omp for schedule(static)
    for (std::size_t i = 0; i < curr_pressure_.size(); ++i)
    {
        curr_pressure_[i] = 0.0f;
        next_pressure_[i] = 0.0f;
    }

    for (int iter = 0; iter < parameters_.pressure_iterations; ++iter)
    {
        #pragma omp for schedule(static)
        for (std::size_t y = 0; y < mesh_.height(); ++y)
        {
            for (std::size_t x = 0; x < mesh_.width(); ++x)
            {
                const auto i = mesh_.index(x, y);
                const auto ti = mesh_.type(i);

                switch (ti)
                {
                    case CellType::Fluid:
                    {
                        float left_pressure  = curr_pressure_[i];
                        float right_pressure = curr_pressure_[i];
                        float up_pressure    = curr_pressure_[i];
                        float down_pressure  = curr_pressure_[i];

                        if (x > 0)
                        {
                            const auto j = mesh_.index(x - 1, y);
                            const auto tj = mesh_.type(j);
                            if (tj == CellType::Fluid)
                            {
                                left_pressure = curr_pressure_[j];
                            }
                        }

                        if (x + 1 < mesh_.width())
                        {
                            const auto j = mesh_.index(x + 1, y);
                            const auto tj = mesh_.type(j);
                            if (tj == CellType::Fluid)
                            {
                                right_pressure = curr_pressure_[j];
                            }
                        }

                        if (y > 0)
                        {
                            const auto j = mesh_.index(x, y - 1);
                            const auto tj = mesh_.type(j);
                            if (tj == CellType::Fluid)
                            {
                                up_pressure = curr_pressure_[j];
                            }
                        }

                        if (y + 1 < mesh_.height())
                        {
                            const auto j = mesh_.index(x, y + 1);
                            const auto tj = mesh_.type(j);
                            if (tj == CellType::Fluid)
                            {
                                down_pressure = curr_pressure_[j];
                            }
                        }

                        next_pressure_[i] =
                            0.25f *
                            (
                                left_pressure +
                                right_pressure +
                                up_pressure +
                                down_pressure -
                                divergence_[i]
                            );

                        break;
                    }

                    default:
                    {
                        next_pressure_[i] = 0.0f;
                        break;
                    }
                }
            }
        }

        #pragma omp single
        std::swap(curr_pressure_, next_pressure_);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Subtract pressure gradient
    ///////////////////////////////////////////////////////////////////////////

    #pragma omp for schedule(static)
    for (std::size_t y = 0; y < mesh_.height(); ++y)
    {
        for (std::size_t x = 0; x < mesh_.width(); ++x)
        {
            const auto i = mesh_.index(x, y);
            const auto ti = mesh_.type(i);

            switch (ti)
            {
                case CellType::Fluid:
                {
                    float left_pressure  = curr_pressure_[i];
                    float right_pressure = curr_pressure_[i];
                    float up_pressure    = curr_pressure_[i];
                    float down_pressure  = curr_pressure_[i];

                    if (x > 0)
                    {
                        const auto j = mesh_.index(x - 1, y);
                        const auto tj = mesh_.type(j);
                        if (tj == CellType::Fluid)
                        {
                            left_pressure = curr_pressure_[j];
                        }
                    }

                    if (x + 1 < mesh_.width())
                    {
                        const auto j = mesh_.index(x + 1, y);
                        const auto tj = mesh_.type(j);
                        if (tj == CellType::Fluid)
                        {
                            right_pressure = curr_pressure_[j];
                        }
                    }

                    if (y > 0)
                    {
                        const auto j = mesh_.index(x, y - 1);
                        const auto tj = mesh_.type(j);
                        if (tj == CellType::Fluid)
                        {
                            up_pressure = curr_pressure_[j];
                        }
                    }

                    if (y + 1 < mesh_.height())
                    {
                        const auto j = mesh_.index(x, y + 1);
                        const auto tj = mesh_.type(j);
                        if (tj == CellType::Fluid)
                        {
                            down_pressure = curr_pressure_[j];
                        }
                    }

                    curr_velocity_[i].x -= 0.5f * (right_pressure - left_pressure);
                    curr_velocity_[i].y -= 0.5f * (down_pressure - up_pressure);

                    break;
                }

                default:
                {
                    break;
                }
            }
        }
    }
}

// Bilinear temperature sampling
float CpuSolver::sample_temperature(float x, float y, float fallback_temperature) const
{
    const auto x0 = static_cast<std::size_t>(std::floor(x));
    const auto y0 = static_cast<std::size_t>(std::floor(y));

    const auto x1 = std::min(x0 + 1, mesh_.width() - 1);
    const auto y1 = std::min(y0 + 1, mesh_.height() - 1);

    const float sx = x - static_cast<float>(x0);
    const float sy = y - static_cast<float>(y0);

    auto read = [&](std::size_t i)
    {
        return mesh_.type(i) == CellType::Fluid
            ? curr_temperature_[i]
            : fallback_temperature;
    };

    const auto t00 = read(mesh_.index(x0, y0));
    const auto t10 = read(mesh_.index(x1, y0));
    const auto t01 = read(mesh_.index(x0, y1));
    const auto t11 = read(mesh_.index(x1, y1));

    const auto a = t00 + sx * (t10 - t00);
    const auto b = t01 + sx * (t11 - t01);

    return a + sy * (b - a);
}

// Bilinear velocity sampling
glm::vec2 CpuSolver::sample_velocity(float x, float y) const
{
    const auto x0 = static_cast<std::size_t>(std::floor(x));
    const auto y0 = static_cast<std::size_t>(std::floor(y));

    const auto x1 = std::min(x0 + 1, mesh_.width() - 1);
    const auto y1 = std::min(y0 + 1, mesh_.height() - 1);

    const float sx = x - static_cast<float>(x0);
    const float sy = y - static_cast<float>(y0);

    auto read = [&](std::size_t i)
    {
        return mesh_.type(i) == CellType::Fluid
            ? curr_velocity_[i]
            : glm::vec2(0.0f, 0.0f);
    };

    const auto v00 = read(mesh_.index(x0, y0));
    const auto v10 = read(mesh_.index(x1, y0));
    const auto v01 = read(mesh_.index(x0, y1));
    const auto v11 = read(mesh_.index(x1, y1));

    const auto a = v00 + sx * (v10 - v00);
    const auto b = v01 + sx * (v11 - v01);

    return a + sy * (b - a);
}
