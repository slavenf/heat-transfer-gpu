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

#include <cstdlib>
#include <cxxopts.hpp>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <SFML/Graphics.hpp>
#include <sstream>
#include <string>

#include "CpuSolver.hpp"
#include "GpuSolverOpenCl.hpp"
#include "GpuSolverOpenGl.hpp"
#include "Mesh.hpp"
#include "SimulationCaseLoader.hpp"

struct CommandLineOptions
{
    std::filesystem::path case_path;
    std::string solver_type;
    int num_iterations_per_frame;
    int scale;
};

static CommandLineOptions parse_command_line_options(int argc, char* argv[])
{
    CommandLineOptions command_line_options;

    cxxopts::Options options("heat-transfer", "Heat transfer simulation");
    options.add_options()
        ("case", "Path to simulation case JSON file", cxxopts::value<std::string>(), "PATH")
        ("s", "Solver type: cpu, opencl, opengl", cxxopts::value<std::string>()->default_value("cpu"), "SOLVER")
        ("i", "Number of solver iterations per rendered frame", cxxopts::value<int>()->default_value("200"), "COUNT")
        ("z", "Window scale factor", cxxopts::value<int>()->default_value("1"), "FACTOR")
        ("h,help", "Print usage");
    options.parse_positional({"case"});
    options.positional_help("CASE");

    try
    {
        const auto result = options.parse(argc, argv);

        if (result.count("help") != 0)
        {
            std::cout << options.help() << "\n";
            std::exit(0);
        }

        if (result.count("case") == 0)
        {
            std::cerr << "ERROR: Missing simulation case path" << "\n\n" << options.help() << "\n";
            std::exit(1);
        }

        command_line_options.case_path = result["case"].as<std::string>();
        command_line_options.solver_type = result["s"].as<std::string>();
        command_line_options.num_iterations_per_frame = result["i"].as<int>();
        command_line_options.scale = result["z"].as<int>();
    }
    catch (const std::exception& error)
    {
        std::cerr << "ERROR: " << error.what() << "\n\n" << options.help() << "\n";
        std::exit(1);
    }

    if
    (
        command_line_options.solver_type != "cpu" &&
        command_line_options.solver_type != "opencl" &&
        command_line_options.solver_type != "opengl")
    {
        std::cerr << "ERROR: Invalid solver type: " << command_line_options.solver_type << ". Valid solvers: cpu, opencl, opengl." << "\n";
        std::exit(1);
    }

    if (command_line_options.num_iterations_per_frame <= 0)
    {
        std::cerr << "ERROR: Number of solver iterations per rendered frame must be greater than 0" << "\n";
        std::exit(1);
    }

    if (command_line_options.scale <= 0)
    {
        std::cerr << "ERROR: Window scale factor must be greater than 0" << "\n";
        std::exit(1);
    }

    return command_line_options;
}

static void center_window(sf::Window& window)
{
    const auto desktop_size = sf::VideoMode::getDesktopMode().size;
    const auto window_size = window.getSize();

    const int x = (desktop_size.x - window_size.x) / 2;
    const int y = (desktop_size.y - window_size.y) / 2;

    window.setPosition({x, y});
}

int main(int argc, char* argv[])
{
    const auto options = parse_command_line_options(argc, argv);

    const SimulationCase simulation_case = load_simulation_case_from_json(options.case_path);
    const Mesh& mesh = simulation_case.mesh;
    const SolverParameters& solver_parameters = simulation_case.solver_parameters;

    sf::ContextSettings context_settings;
    context_settings.depthBits = 0;
    context_settings.stencilBits = 0;
    context_settings.antiAliasingLevel = 0;
    context_settings.majorVersion = 4;
    context_settings.minorVersion = 3;
    context_settings.attributeFlags = sf::ContextSettings::Default;

    sf::RenderWindow window
    (
        sf::VideoMode
        (
            sf::Vector2u
            (
                mesh.width() * options.scale,
                mesh.height() * options.scale
            )
        ),
        "Heat Transfer",
        sf::Style::Titlebar | sf::Style::Close,
        sf::State::Windowed,
        context_settings
    );

    const sf::View simulation_view
    (
        sf::FloatRect
        (
            {0.0f, 0.0f},
            {static_cast<float>(mesh.width()), static_cast<float>(mesh.height())}
        )
    );

    window.setView(simulation_view);

    center_window(window);

    std::unique_ptr<Solver> solver;

    if (options.solver_type == "cpu")
    {
        solver = std::make_unique<CpuSolver>(mesh, solver_parameters);
    }
    else if (options.solver_type == "opencl")
    {
        solver = std::make_unique<GpuSolverOpenCl>(mesh, solver_parameters);
    }
    else if (options.solver_type == "opengl")
    {
        solver = std::make_unique<GpuSolverOpenGl>(mesh, solver_parameters, window);
    }
    else
    {
        throw std::runtime_error("Unhandled solver type: " + options.solver_type);
    }

    sf::Font font("assets/fonts/RobotoCondensed-Regular.ttf");

    sf::Text text(font);
    text.setCharacterSize(14);
    text.setFillColor(sf::Color::White);
    text.setPosition({5.0f, 0.0f});

    sf::Clock clock;

    bool show_hud = true;
    bool show_velocity_field = false;

    while (window.isOpen())
    {
        const float dt = clock.restart().asSeconds();

        while (const auto event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
            {
                window.close();
            }
            else if (const auto* key_pressed = event->getIf<sf::Event::KeyPressed>())
            {
                if (key_pressed->scancode == sf::Keyboard::Scancode::Escape)
                {
                    window.close();
                }
                else if (key_pressed->scancode == sf::Keyboard::Scancode::R)
                {
                    solver->reset();
                    clock.restart();
                }
                else if (key_pressed->scancode == sf::Keyboard::Scancode::H)
                {
                    show_hud = !show_hud;
                }
                else if (key_pressed->scancode == sf::Keyboard::Scancode::V)
                {
                    show_velocity_field = !show_velocity_field;
                }
            }
        }

        solver->step(options.num_iterations_per_frame);

        window.clear();

        window.setView(simulation_view);

        solver->draw(window);

        if (show_velocity_field)
        {
            solver->draw_velocity_field(window);
        }

        window.setView(window.getDefaultView());

        if (show_hud)
        {
            // Update HUD
            {
                std::ostringstream ss;
                ss << int(options.num_iterations_per_frame / dt) << " iter/s" << "\n";
                ss << "Avg T: " << solver->average_temperature() << "\n";
                ss << "Max disp: " << solver->max_displacement() << "\n";
                text.setString(ss.str());
            }

            window.draw(text);
        }

        window.display();
    }
}
