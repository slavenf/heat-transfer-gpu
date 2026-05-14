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

#version 430

in vec2 v_uv;

out vec4 frag_color;

layout(binding = 0)
uniform sampler2D curr_;

layout(binding = 1)
uniform usampler2D type_;

uniform float min_temp_;

uniform float max_temp_;

uniform ivec2 size_;

#define CELL_TYPE_SOLID     0u
#define CELL_TYPE_SOURCE    1u
#define CELL_TYPE_VACUUM    2u
#define CELL_TYPE_FLUID     3u

uint cell_type(ivec2 p)
{
    return texelFetch(type_, p, 0).r;
}

float curr_temp(ivec2 p)
{
    return texelFetch(curr_, p, 0).r;
}

vec3 temperature_to_color(float temp)
{
    // blue -> cyan -> green -> yellow -> red gradient

    const float u = clamp((temp - min_temp_) / (max_temp_ - min_temp_), 0.0, 1.0);

    // blue -> cyan
    if (u < 0.25)
    {
        const float t = u / 0.25;
        return vec3(0.0, t, 1.0);
    }

    // cyan -> green
    if (u < 0.50)
    {
        const float t = (u - 0.25) / 0.25;
        return vec3(0.0, 1.0, 1.0 - t);
    }

    // green -> yellow
    if (u < 0.75)
    {
        const float t = (u - 0.50) / 0.25;
        return vec3(t, 1.0, 0.0);
    }

    // yellow -> red
    const float t = (u - 0.75) / 0.25;
    return vec3(1.0, 1.0 - t, 0.0);
}

void main()
{
    const vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
    const ivec2 p = ivec2(clamp(uv, vec2(0.0), vec2(0.999999)) * vec2(size_));

    switch (cell_type(p))
    {
        case CELL_TYPE_VACUUM:
        {
            frag_color = vec4(0.0, 0.0, 0.0, 1.0);
            break;
        }

        case CELL_TYPE_SOURCE:
        {
            frag_color = vec4(1.0, 1.0, 1.0, 1.0);
            break;
        }

        case CELL_TYPE_SOLID:
        case CELL_TYPE_FLUID:
        {
            frag_color = vec4(temperature_to_color(curr_temp(p)), 1.0);
            break;
        }
    }
}
