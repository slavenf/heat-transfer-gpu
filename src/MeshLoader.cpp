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

#include "MeshLoader.hpp"

Mesh load_hardcoded_mesh1()
{
    const std::size_t width = 400;
    const std::size_t height = 200;

    std::vector<CellType> type(width * height, CellType::Vacuum);

    auto index = [width](std::size_t x, std::size_t y)
    {
        return y * width + x;
    };

    // Metal plate
    for (std::size_t y = 0; y < height; ++y)
    {
        for (std::size_t x = 0; x < width; ++x)
        {
            type[index(x, y)] = CellType::Metal;
        }
    }

    // Slit in metal plate
    for (std::size_t y = 50; y < 150; ++y)
    {
        for (std::size_t x = 200; x < 210; ++x)
        {
            type[index(x, y)] = CellType::Vacuum;
        }
    }

    // Fixed-temperature source
    for (std::size_t y = 95; y < 105; ++y)
    {
        for (std::size_t x = 100; x < 110; ++x)
        {
            type[index(x, y)] = CellType::Source;
        }
    }

    return Mesh(width, height, std::move(type));
}

Mesh load_hardcoded_mesh2()
{
    const std::size_t width = 300;
    const std::size_t height = 400;

    std::vector<CellType> type(width * height, CellType::Vacuum);

    auto index = [width](std::size_t x, std::size_t y)
    {
        return y * width + x;
    };

    for (std::size_t y = 0; y < height; ++y)
    {
        for (std::size_t x = 0; x < width; ++x)
        {
            type[index(x, y)] = CellType::Vacuum;
        }
    }

    for (std::size_t y = 20; y < 380; ++y)
    {
        for (std::size_t x = 20; x < 280; ++x)
        {
            type[index(x, y)] = CellType::Metal;
        }
    }

    for (std::size_t y = 250; y < 260; ++y)
    {
        for (std::size_t x = 20; x < 125; ++x)
        {
            type[index(x, y)] = CellType::Vacuum;
        }
    }

    for (std::size_t y = 250; y < 260; ++y)
    {
        for (std::size_t x = 175; x < 280; ++x)
        {
            type[index(x, y)] = CellType::Vacuum;
        }
    }

    for (std::size_t y = 300; y < 380; ++y)
    {
        for (std::size_t x = 115; x < 125; ++x)
        {
            type[index(x, y)] = CellType::Vacuum;
        }
    }

    for (std::size_t y = 300; y < 380; ++y)
    {
        for (std::size_t x = 200; x < 210; ++x)
        {
            type[index(x, y)] = CellType::Vacuum;
        }
    }

    for (std::size_t y = 20; y < 40; ++y)
    {
        for (std::size_t x = 115; x < 125; ++x)
        {
            type[index(x, y)] = CellType::Vacuum;
        }
    }

    for (std::size_t y = 80; y < 90; ++y)
    {
        for (std::size_t x = 20; x < 125; ++x)
        {
            type[index(x, y)] = CellType::Vacuum;
        }
    }

    for (std::size_t y = 130; y < 250; ++y)
    {
        for (std::size_t x = 115; x < 125; ++x)
        {
            type[index(x, y)] = CellType::Vacuum;
        }
    }

    for (std::size_t y = 265; y < 275; ++y)
    {
        for (std::size_t x = 250; x < 260; ++x)
        {
            type[index(x, y)] = CellType::Source;
        }
    }

    return Mesh(width, height, std::move(type));
}
