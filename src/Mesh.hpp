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

#ifndef FILE_MESH_HPP_INCLUDED
#define FILE_MESH_HPP_INCLUDED

#include <cstddef>
#include <cstdint>
#include <vector>

enum class CellType : std::uint8_t
{
    Vacuum,
    Metal,
    Source
};

class Mesh
{
public:

    Mesh(std::size_t width, std::size_t height, std::vector<CellType> type)
        : width_(width)
        , height_(height)
        , type_(std::move(type))
    {}

    std::size_t width() const noexcept
    {
        return width_;
    }

    std::size_t height() const noexcept
    {
        return height_;
    }

    std::size_t index(std::size_t x, std::size_t y) const noexcept
    {
        return y * width_ + x;
    }

    CellType type(std::size_t i) const noexcept
    {
        return type_[i];
    }

    bool is_conductive(std::size_t i) const noexcept
    {
        const auto t = type_[i];
        return t == CellType::Metal || t == CellType::Source;
    }

private:

    std::size_t width_;
    std::size_t height_;
    std::vector<CellType> type_;
};

#endif // FILE_MESH_HPP_INCLUDED
