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

#include "temperature_to_color.hpp"

#include <algorithm>
#include <array>

sf::Color temperature_to_color(float temp, float min_temp, float max_temp)
{
    #if 0 // blue -> red gradient

    const float u = std::clamp((temp - min_temp) / (max_temp - min_temp), 0.0f, 1.0f);

    std::uint8_t r = static_cast<std::uint8_t>(255.f * u);
    std::uint8_t b = static_cast<std::uint8_t>(255.f * (1.f - u));

    return sf::Color(r, 0, b);

    #endif

    ///////////////////////////////////////////////////////////////////////////

    #if 0 // blue -> cyan -> yellow -> red gradient

    const float u = std::clamp((temp - min_temp) / (max_temp - min_temp), 0.0f, 1.0f);

    // blue -> cyan
    if (u < 0.33f)
    {
        const float t = u / 0.33f;
        return sf::Color
        (
            static_cast<std::uint8_t>(0.f),
            static_cast<std::uint8_t>(255.f * t),
            static_cast<std::uint8_t>(255.f)
        );
    }

    // cyan -> yellow
    if (u < 0.66f)
    {
        const float t = (u - 0.33f) / 0.33f;
        return sf::Color
        (
            static_cast<std::uint8_t>(255.f * t),
            static_cast<std::uint8_t>(255.f),
            static_cast<std::uint8_t>(255.f * (1.f - t))
        );
    }

    // yellow -> red
    const float t = (u - 0.66f) / 0.34f;
    return sf::Color
    (
        static_cast<std::uint8_t>(255.f),
        static_cast<std::uint8_t>(255.f * (1.f - t)),
        static_cast<std::uint8_t>(0.f)
    );

    #endif

    ///////////////////////////////////////////////////////////////////////////

    #if 1 // blue -> cyan -> green -> yellow -> red gradient

    const float u = std::clamp((temp - min_temp) / (max_temp - min_temp), 0.0f, 1.0f);

    auto lerp = [](std::uint8_t a, std::uint8_t b, float t) -> std::uint8_t
    {
        return static_cast<std::uint8_t>(a + (b - a) * t);
    };

    // blue -> cyan
    if (u < 0.25f)
    {
        const float t = u / 0.25f;
        return sf::Color
        (
            0,
            lerp(0, 255, t),
            255
        );
    }

    // cyan -> green
    if (u < 0.50f)
    {
        const float t = (u - 0.25f) / 0.25f;
        return sf::Color
        (
            0,
            255,
            lerp(255, 0, t)
        );
    }

    // green -> yellow
    if (u < 0.75f)
    {
        const float t = (u - 0.50f) / 0.25f;
        return sf::Color
        (
            lerp(0, 255, t),
            255,
            0
        );
    }

    // yellow -> red
    const float t = (u - 0.75f) / 0.25f;
    return sf::Color
    (
        255,
        lerp(255, 0, t),
        0
    );

    #endif

    ///////////////////////////////////////////////////////////////////////////

    #if 0 // blue -> cyan -> green -> yellow -> orange -> red -> white

    const float u = std::clamp((temp - min_temp) / (max_temp - min_temp), 0.0f, 1.0f);

    auto lerp = [](std::uint8_t a, std::uint8_t b, float t) -> std::uint8_t
    {
        return static_cast<std::uint8_t>(a + (b - a) * t);
    };

    auto mix = [&](sf::Color c0, sf::Color c1, float t) -> sf::Color
    {
        return sf::Color
        (
            lerp(c0.r, c1.r, t),
            lerp(c0.g, c1.g, t),
            lerp(c0.b, c1.b, t)
        );
    };

    constexpr sf::Color blue  {  0,   0, 255};
    constexpr sf::Color cyan  {  0, 255, 255};
    constexpr sf::Color green {  0, 255,   0};
    constexpr sf::Color yellow{255, 255,   0};
    constexpr sf::Color orange{255, 165,   0};
    constexpr sf::Color red   {255,   0,   0};
    constexpr sf::Color white {255, 255, 255};

    // 6 equal segments across [0,1]

    if (u < 1.0f / 6.0f)
    {
        float t = u * 6.0f;
        return mix(blue, cyan, t);
    }

    if (u < 2.0f / 6.0f)
    {
        float t = (u - 1.0f / 6.0f) * 6.0f;
        return mix(cyan, green, t);
    }

    if (u < 3.0f / 6.0f)
    {
        float t = (u - 2.0f / 6.0f) * 6.0f;
        return mix(green, yellow, t);
    }

    if (u < 4.0f / 6.0f)
    {
        float t = (u - 3.0f / 6.0f) * 6.0f;
        return mix(yellow, orange, t);
    }

    if (u < 5.0f / 6.0f)
    {
        float t = (u - 4.0f / 6.0f) * 6.0f;
        return mix(orange, red, t);
    }

    float t = (u - 5.0f / 6.0f) * 6.0f;
    return mix(red, white, t);

    #endif

    ///////////////////////////////////////////////////////////////////////////

    #if 0 // inferno

    const float u = std::clamp((temp - min_temp) / (max_temp - min_temp), 0.0f, 1.0f);

    struct RGB
    {
        float r;
        float g;
        float b;
    };

    // Compact Inferno-style control points.
    // These are not the full 256-sample table, but a good approximation.
    constexpr std::array<RGB, 8> colors
    {{
        {0.001f, 0.000f, 0.014f}, // near black
        {0.087f, 0.044f, 0.224f}, // dark purple
        {0.258f, 0.039f, 0.406f}, // purple
        {0.478f, 0.016f, 0.318f}, // magenta-red
        {0.709f, 0.112f, 0.110f}, // red-orange
        {0.902f, 0.364f, 0.047f}, // orange
        {0.987f, 0.645f, 0.039f}, // yellow-orange
        {0.988f, 0.998f, 0.645f}  // pale yellow
    }};

    const float scaled = u * static_cast<float>(colors.size() - 1);
    const std::size_t i0 = static_cast<std::size_t>(scaled);
    const std::size_t i1 = std::min(i0 + 1, colors.size() - 1);
    const float t = scaled - static_cast<float>(i0);

    const auto lerp = [](float a, float b, float x) -> float
    {
        return a + (b - a) * x;
    };

    const float r = lerp(colors[i0].r, colors[i1].r, t);
    const float g = lerp(colors[i0].g, colors[i1].g, t);
    const float b = lerp(colors[i0].b, colors[i1].b, t);

    return sf::Color
    (
        static_cast<std::uint8_t>(255.0f * std::clamp(r, 0.0f, 1.0f)),
        static_cast<std::uint8_t>(255.0f * std::clamp(g, 0.0f, 1.0f)),
        static_cast<std::uint8_t>(255.0f * std::clamp(b, 0.0f, 1.0f))
    );

    #endif
}
