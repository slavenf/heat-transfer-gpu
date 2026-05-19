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

#include "SimulationCaseLoader.hpp"

#include <cstddef>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static void from_json(const nlohmann::json& data, CellType& type)
{
    const std::string type_as_string = data.get<std::string>();

    if (type_as_string == "solid")
    {
        type = CellType::Solid;
        return;
    }

    if (type_as_string == "source")
    {
        type = CellType::Source;
        return;
    }

    if (type_as_string == "vacuum")
    {
        type = CellType::Vacuum;
        return;
    }

    if (type_as_string == "fluid")
    {
        type = CellType::Fluid;
        return;
    }

    throw std::runtime_error("Unknown cell type: " + type_as_string);
}

static void parse_rectangle_region
(
    const nlohmann::json& region,
    const std::size_t mesh_width,
    const std::size_t mesh_height,
    std::vector<CellType>& mesh_cell_type,
    std::vector<Real>& mesh_initial_temperature
)
{
    const CellType region_cell_type = region["cell"];

    const std::size_t x0 = region["x"];
    const std::size_t y0 = region["y"];

    const std::size_t width = region["width"];
    const std::size_t height = region["height"];

    if (x0 + width > mesh_width || y0 + height > mesh_height)
    {
        throw std::runtime_error("Rect region extends outside the mesh");
    }

    const Real initial_temperature = region["initial_temperature"];

    Real temperature_noise_min = Real(0.0);
    Real temperature_noise_max = Real(0.0);

    if (region.contains("temperature_noise"))
    {
        temperature_noise_min = region["temperature_noise"]["min"];
        temperature_noise_max = region["temperature_noise"]["max"];
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Real> temperature_noise_dist(temperature_noise_min, temperature_noise_max);

    for (std::size_t y = y0; y < y0 + height; ++y)
    {
        for (std::size_t x = x0; x < x0 + width; ++x)
        {
            const std::size_t i = y * mesh_width + x;

            mesh_cell_type[i] = region_cell_type;

            mesh_initial_temperature[i] = initial_temperature + temperature_noise_dist(gen);
        }
    }
}

static Mesh parse_mesh(const nlohmann::json& data)
{
    const std::size_t width = data["width"];
    const std::size_t height = data["height"];

    const Real min_temperature = data["min_temperature"];
    const Real max_temperature = data["max_temperature"];

    const CellType default_cell_type = data["default_cell"]["type"];
    const Real default_initial_temperature = data["default_cell"]["initial_temperature"];

    std::vector<CellType> cell_type(width * height, default_cell_type);
    std::vector<Real> initial_temperature(width * height, default_initial_temperature);

    for (const auto& region : data["regions"])
    {
        const std::string region_type = region["type"];

        if (region_type == "rectangle")
        {
            parse_rectangle_region
            (
                region,
                width,
                height,
                cell_type,
                initial_temperature
            );
        }
        else
        {
            throw std::runtime_error("Unknown region type: " + region_type);
        }
    }

    return Mesh
    (
        width,
        height,
        std::move(cell_type),
        std::move(initial_temperature),
        min_temperature,
        max_temperature
    );
}

static SolverParameters parse_solver_parameters(const nlohmann::json& data)
{
    SolverParameters parameters;

    parameters.dt = data["dt"];

    parameters.buoyancy = data["buoyancy"];

    parameters.thermal_diffusion = data["thermal_diffusion"];

    parameters.velocity_damping = data["velocity_damping"];

    parameters.viscosity = data["viscosity"];

    parameters.pressure_iterations = data["pressure_iterations"];

    parameters.source_heat_transfer = data["source_heat_transfer"];

    return parameters;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

SimulationCase load_simulation_case_from_json(const std::filesystem::path& path)
{
    std::ifstream ifs(path);

    if (!ifs)
    {
        throw std::runtime_error("Cannot open simulation case: " + path.string());
    }

    const nlohmann::json data = nlohmann::json::parse(ifs, nullptr, true, true);

    return SimulationCase
    (
        parse_mesh(data["mesh"]),
        parse_solver_parameters(data["solver_parameters"])
    );
}
