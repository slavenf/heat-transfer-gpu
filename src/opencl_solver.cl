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

#define CELL_TYPE_VACUUM    0u
#define CELL_TYPE_METAL     1u
#define CELL_TYPE_SOURCE    2u

bool is_conductive(__global const uchar* type, uint i)
{
    const uchar t = type[i];
    return t == CELL_TYPE_METAL || t == CELL_TYPE_SOURCE;
}

__kernel void step_heat
(
    __global const uchar* type,
    __global const float* curr,
    __global float* next,
    uint width,
    uint height,
    float diffusion
)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if (x >= width || y >= height)
    {
        return;
    }

    const uint i = y * width + x;

    switch (type[i])
    {
        case CELL_TYPE_VACUUM:
        case CELL_TYPE_SOURCE:
        {
            next[i] = curr[i];
            break;
        }

        case CELL_TYPE_METAL:
        {
            const float center = curr[i];

            float sum = 0.0f;

            if (x > 0)
            {
                const uint j = y * width + x - 1;
                if (is_conductive(type, j))
                {
                    sum += curr[j] - center;
                }
            }

            if (x + 1 < width)
            {
                const uint j = y * width + x + 1;
                if (is_conductive(type, j))
                {
                    sum += curr[j] - center;
                }
            }

            if (y > 0)
            {
                const uint j = (y - 1) * width + x;
                if (is_conductive(type, j))
                {
                    sum += curr[j] - center;
                }
            }

            if (y + 1 < height)
            {
                const uint j = (y + 1) * width + x;
                if (is_conductive(type, j))
                {
                    sum += curr[j] - center;
                }
            }

            next[i] = center + diffusion * sum;
            break;
        }
    }
}

uchar lerp_uchar(uchar a, uchar b, float t)
{
    return convert_uchar_sat((float)a + ((float)b - (float)a) * t);
}

uchar4 temperature_to_color(float temp, float min_temp, float max_temp)
{
    // blue -> cyan -> green -> yellow -> red gradient

    const float u = clamp((temp - min_temp) / (max_temp - min_temp), 0.0f, 1.0f);

    // blue -> cyan
    if (u < 0.25f)
    {
        const float t = u / 0.25f;
        return (uchar4)(0, lerp_uchar(0, 255, t), 255, 255);
    }

    // cyan -> green
    if (u < 0.50f)
    {
        const float t = (u - 0.25f) / 0.25f;
        return (uchar4)(0, 255, lerp_uchar(255, 0, t), 255);
    }

    // green -> yellow
    if (u < 0.75f)
    {
        const float t = (u - 0.50f) / 0.25f;
        return (uchar4)(lerp_uchar(0, 255, t), 255, 0, 255);
    }

    // yellow -> red
    const float t = (u - 0.75f) / 0.25f;
    return (uchar4)(255, lerp_uchar(255, 0, t), 0, 255);
}

__kernel void render_heat
(
    __global const uchar* type,
    __global const float* curr,
    __global uchar4* pixels,
    uint width,
    uint height,
    float min_temp,
    float max_temp
)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if (x >= width || y >= height)
    {
        return;
    }

    const uint i = y * width + x;

    switch (type[i])
    {
        case CELL_TYPE_VACUUM:
        {
            pixels[i] = (uchar4)(0, 0, 0, 255);
            break;
        }

        case CELL_TYPE_SOURCE:
        {
            pixels[i] = (uchar4)(255, 255, 255, 255);
            break;
        }

        case CELL_TYPE_METAL:
        {
            pixels[i] = temperature_to_color(curr[i], min_temp, max_temp);
            break;
        }
    }
}
