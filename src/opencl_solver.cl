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

#define CELL_TYPE_SOLID     0u
#define CELL_TYPE_SOURCE    1u
#define CELL_TYPE_VACUUM    2u
#define CELL_TYPE_FLUID     3u

uint index_of(uint x, uint y, uint width)
{
    return y * width + x;
}

float sample_temperature
(
    __global const uchar* type,
    __global const float* temperature,
    float x,
    float y,
    uint width,
    uint height,
    float fallback
)
{
    const uint x0 = (uint)floor(x);
    const uint y0 = (uint)floor(y);
    const uint x1 = min(x0 + 1u, width - 1u);
    const uint y1 = min(y0 + 1u, height - 1u);

    const float sx = x - (float)x0;
    const float sy = y - (float)y0;

    const uint i00 = index_of(x0, y0, width);
    const uint i10 = index_of(x1, y0, width);
    const uint i01 = index_of(x0, y1, width);
    const uint i11 = index_of(x1, y1, width);

    const float t00 = type[i00] == CELL_TYPE_FLUID ? temperature[i00] : fallback;
    const float t10 = type[i10] == CELL_TYPE_FLUID ? temperature[i10] : fallback;
    const float t01 = type[i01] == CELL_TYPE_FLUID ? temperature[i01] : fallback;
    const float t11 = type[i11] == CELL_TYPE_FLUID ? temperature[i11] : fallback;

    const float a = t00 + sx * (t10 - t00);
    const float b = t01 + sx * (t11 - t01);

    return a + sy * (b - a);
}

float2 sample_velocity
(
    __global const uchar* type,
    __global const float2* velocity,
    float x,
    float y,
    uint width,
    uint height
)
{
    const uint x0 = (uint)floor(x);
    const uint y0 = (uint)floor(y);
    const uint x1 = min(x0 + 1u, width - 1u);
    const uint y1 = min(y0 + 1u, height - 1u);

    const float sx = x - (float)x0;
    const float sy = y - (float)y0;

    const uint i00 = index_of(x0, y0, width);
    const uint i10 = index_of(x1, y0, width);
    const uint i01 = index_of(x0, y1, width);
    const uint i11 = index_of(x1, y1, width);

    const float2 v00 = type[i00] == CELL_TYPE_FLUID ? velocity[i00] : (float2)(0.0f, 0.0f);
    const float2 v10 = type[i10] == CELL_TYPE_FLUID ? velocity[i10] : (float2)(0.0f, 0.0f);
    const float2 v01 = type[i01] == CELL_TYPE_FLUID ? velocity[i01] : (float2)(0.0f, 0.0f);
    const float2 v11 = type[i11] == CELL_TYPE_FLUID ? velocity[i11] : (float2)(0.0f, 0.0f);

    const float2 a = v00 + sx * (v10 - v00);
    const float2 b = v01 + sx * (v11 - v01);

    return a + sy * (b - a);
}

__kernel void add_heat
(
    __global const uchar* type,
    __global float* temperature,
    uint width,
    uint height,
    float source_heat_transfer,
    float dt,
    float max_temperature
)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if (x >= width || y >= height)
    {
        return;
    }

    const uint i = index_of(x, y, width);
    const uchar ti = type[i];

    switch (ti)
    {
        case CELL_TYPE_SOLID:
        case CELL_TYPE_FLUID:
        {
            bool touches_source = false;

            if (x > 0u)
            {
                const uint j = index_of(x - 1u, y, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_SOURCE)
                {
                    touches_source = true;
                }
            }

            if (x + 1u < width)
            {
                const uint j = index_of(x + 1u, y, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_SOURCE)
                {
                    touches_source = true;
                }
            }

            if (y > 0u)
            {
                const uint j = index_of(x, y - 1u, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_SOURCE)
                {
                    touches_source = true;
                }
            }

            if (y + 1u < height)
            {
                const uint j = index_of(x, y + 1u, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_SOURCE)
                {
                    touches_source = true;
                }
            }

            if (touches_source)
            {
                temperature[i] += source_heat_transfer * dt;

                if (temperature[i] > max_temperature)
                {
                    temperature[i] = max_temperature;
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

__kernel void add_buoyancy
(
    __global const uchar* type,
    __global const float* temperature,
    __global float2* velocity,
    uint width,
    uint height,
    float ambient_temperature,
    float buoyancy,
    float dt
)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if (x >= width || y >= height)
    {
        return;
    }

    const uint i = index_of(x, y, width);
    const uchar ti = type[i];

    switch (ti)
    {
        case CELL_TYPE_FLUID:
        {
            velocity[i].y -= buoyancy * (temperature[i] - ambient_temperature) * dt;
            break;
        }

        default:
            break;
    }
}

__kernel void apply_velocity_boundaries
(
    __global const uchar* type,
    __global float2* velocity,
    uint width,
    uint height
)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if (x >= width || y >= height)
    {
        return;
    }

    const uint i = index_of(x, y, width);
    const uchar ti = type[i];

    switch (ti)
    {
        case CELL_TYPE_FLUID:
        {
            if (x > 0u)
            {
                const uint j = index_of(x - 1u, y, width);
                const uchar tj = type[j];
                if (tj != CELL_TYPE_FLUID && velocity[i].x < 0.0f)
                {
                    velocity[i].x = 0.0f;
                }
            }
            else if (velocity[i].x < 0.0f)
            {
                velocity[i].x = 0.0f;
            }

            if (x + 1u < width)
            {
                const uint j = index_of(x + 1u, y, width);
                const uchar tj = type[j];
                if (tj != CELL_TYPE_FLUID && velocity[i].x > 0.0f)
                {
                    velocity[i].x = 0.0f;
                }
            }
            else if (velocity[i].x > 0.0f)
            {
                velocity[i].x = 0.0f;
            }

            if (y > 0u)
            {
                const uint j = index_of(x, y - 1u, width);
                const uchar tj = type[j];
                if (tj != CELL_TYPE_FLUID && velocity[i].y < 0.0f)
                {
                    velocity[i].y = 0.0f;
                }
            }
            else if (velocity[i].y < 0.0f)
            {
                velocity[i].y = 0.0f;
            }

            if (y + 1u < height)
            {
                const uint j = index_of(x, y + 1u, width);
                const uchar tj = type[j];
                if (tj != CELL_TYPE_FLUID && velocity[i].y > 0.0f)
                {
                    velocity[i].y = 0.0f;
                }
            }
            else if (velocity[i].y > 0.0f)
            {
                velocity[i].y = 0.0f;
            }

            break;
        }

        default:
        {
            velocity[i] = (float2)(0.0f, 0.0f);
            break;
        }
    }
}

__kernel void advect_velocity
(
    __global const uchar* type,
    __global const float2* curr,
    __global float2* next,
    uint width,
    uint height,
    float dt,
    float damping
)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if (x >= width || y >= height)
    {
        return;
    }

    const uint i = index_of(x, y, width);
    const uchar ti = type[i];

    switch (ti)
    {
        case CELL_TYPE_FLUID:
        {
            const float old_x = clamp((float)x - curr[i].x * dt, 0.0f, (float)(width - 1u));
            const float old_y = clamp((float)y - curr[i].y * dt, 0.0f, (float)(height - 1u));

            next[i] = sample_velocity(type, curr, old_x, old_y, width, height);
            next[i] *= damping;

            break;
        }

        default:
        {
            next[i] = (float2)(0.0f, 0.0f);
            break;
        }
    }
}

__kernel void diffuse_velocity
(
    __global const uchar* type,
    __global const float2* curr,
    __global float2* next,
    uint width,
    uint height,
    float viscosity_dt
)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if (x >= width || y >= height)
    {
        return;
    }

    const uint i = index_of(x, y, width);
    const uchar ti = type[i];

    switch (ti)
    {
        case CELL_TYPE_FLUID:
        {
            float2 sum = (float2)(0.0f, 0.0f);

            if (x > 0u)
            {
                const uint j = index_of(x - 1u, y, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_FLUID)
                {
                    sum += curr[j] - curr[i];
                }
            }

            if (x + 1u < width)
            {
                const uint j = index_of(x + 1u, y, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_FLUID)
                {
                    sum += curr[j] - curr[i];
                }
            }

            if (y > 0u)
            {
                const uint j = index_of(x, y - 1u, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_FLUID)
                {
                    sum += curr[j] - curr[i];
                }
            }

            if (y + 1u < height)
            {
                const uint j = index_of(x, y + 1u, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_FLUID)
                {
                    sum += curr[j] - curr[i];
                }
            }

            next[i] = curr[i] + sum * viscosity_dt;

            break;
        }

        default:
        {
            next[i] = (float2)(0.0f, 0.0f);
            break;
        }
    }
}

__kernel void compute_divergence
(
    __global const uchar* type,
    __global const float2* velocity,
    __global float* divergence,
    uint width,
    uint height
)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if (x >= width || y >= height)
    {
        return;
    }

    const uint i = index_of(x, y, width);
    const uchar ti = type[i];

    switch (ti)
    {
        case CELL_TYPE_FLUID:
        {
            float2 left_velocity = (float2)(0.0f, 0.0f);
            float2 right_velocity = (float2)(0.0f, 0.0f);
            float2 up_velocity = (float2)(0.0f, 0.0f);
            float2 down_velocity = (float2)(0.0f, 0.0f);

            if (x > 0u)
            {
                const uint j = index_of(x - 1u, y, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_FLUID)
                {
                    left_velocity = velocity[j];
                }
            }

            if (x + 1u < width)
            {
                const uint j = index_of(x + 1u, y, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_FLUID)
                {
                    right_velocity = velocity[j];
                }
            }

            if (y > 0u)
            {
                const uint j = index_of(x, y - 1u, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_FLUID)
                {
                    up_velocity = velocity[j];
                }
            }

            if (y + 1u < height)
            {
                const uint j = index_of(x, y + 1u, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_FLUID)
                {
                    down_velocity = velocity[j];
                }
            }

            divergence[i] = 0.5f * (right_velocity.x - left_velocity.x + down_velocity.y - up_velocity.y);

            break;
        }

        default:
        {
            divergence[i] = 0.0f;
            break;
        }
    }
}

__kernel void clear_pressure
(
    __global float* curr,
    __global float* next,
    uint width,
    uint height
)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if (x >= width || y >= height)
    {
        return;
    }

    const uint i = index_of(x, y, width);
    curr[i] = 0.0f;
    next[i] = 0.0f;
}

__kernel void solve_pressure
(
    __global const uchar* type,
    __global const float* curr,
    __global float* next,
    __global const float* divergence,
    uint width,
    uint height
)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if (x >= width || y >= height)
    {
        return;
    }

    const uint i = index_of(x, y, width);
    const uchar ti = type[i];

    switch (ti)
    {
        case CELL_TYPE_FLUID:
        {
            float left_pressure = curr[i];
            float right_pressure = curr[i];
            float up_pressure = curr[i];
            float down_pressure = curr[i];

            if (x > 0u)
            {
                const uint j = index_of(x - 1u, y, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_FLUID)
                {
                    left_pressure = curr[j];
                }
            }

            if (x + 1u < width)
            {
                const uint j = index_of(x + 1u, y, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_FLUID)
                {
                    right_pressure = curr[j];
                }
            }

            if (y > 0u)
            {
                const uint j = index_of(x, y - 1u, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_FLUID)
                {
                    up_pressure = curr[j];
                }
            }

            if (y + 1u < height)
            {
                const uint j = index_of(x, y + 1u, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_FLUID)
                {
                    down_pressure = curr[j];
                }
            }

            next[i] = 0.25f * (left_pressure + right_pressure + up_pressure + down_pressure - divergence[i]);

            break;
        }

        default:
        {
            next[i] = 0.0f;
            break;
        }
    }
}

__kernel void subtract_pressure_gradient
(
    __global const uchar* type,
    __global const float* pressure,
    __global float2* velocity,
    uint width,
    uint height
)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if (x >= width || y >= height)
    {
        return;
    }

    const uint i = index_of(x, y, width);
    const uchar ti = type[i];

    switch (ti)
    {
        case CELL_TYPE_FLUID:
        {
            float left_pressure = pressure[i];
            float right_pressure = pressure[i];
            float up_pressure = pressure[i];
            float down_pressure = pressure[i];

            if (x > 0u)
            {
                const uint j = index_of(x - 1u, y, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_FLUID)
                {
                    left_pressure = pressure[j];
                }
            }

            if (x + 1u < width)
            {
                const uint j = index_of(x + 1u, y, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_FLUID)
                {
                    right_pressure = pressure[j];
                }
            }

            if (y > 0u)
            {
                const uint j = index_of(x, y - 1u, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_FLUID)
                {
                    up_pressure = pressure[j];
                }
            }

            if (y + 1u < height)
            {
                const uint j = index_of(x, y + 1u, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_FLUID)
                {
                    down_pressure = pressure[j];
                }
            }

            velocity[i].x -= 0.5f * (right_pressure - left_pressure);
            velocity[i].y -= 0.5f * (down_pressure - up_pressure);

            break;
        }

        default:
        {
            break;
        }
    }
}

__kernel void advect_temperature
(
    __global const uchar* type,
    __global const float* curr,
    __global float* next,
    __global const float2* velocity,
    uint width,
    uint height,
    float dt
)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if (x >= width || y >= height)
    {
        return;
    }

    const uint i = index_of(x, y, width);
    const uchar ti = type[i];

    switch (ti)
    {
        case CELL_TYPE_FLUID:
        {
            const float old_x = clamp((float)x - velocity[i].x * dt, 0.0f, (float)(width - 1u));
            const float old_y = clamp((float)y - velocity[i].y * dt, 0.0f, (float)(height - 1u));

            next[i] = sample_temperature(type, curr, old_x, old_y, width, height, curr[i]);

            break;
        }

        default:
        {
            next[i] = curr[i];
            break;
        }
    }
}

__kernel void diffuse_temperature
(
    __global const uchar* type,
    __global const float* curr,
    __global float* next,
    uint width,
    uint height,
    float diffusion_dt
)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if (x >= width || y >= height)
    {
        return;
    }

    const uint i = index_of(x, y, width);
    const uchar ti = type[i];

    switch (ti)
    {
        case CELL_TYPE_SOLID:
        case CELL_TYPE_FLUID:
        {
            float sum = 0.0f;

            if (x > 0u)
            {
                const uint j = index_of(x - 1u, y, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_SOLID || tj == CELL_TYPE_FLUID || tj == CELL_TYPE_SOURCE)
                {
                    sum += curr[j] - curr[i];
                }
            }

            if (x + 1u < width)
            {
                const uint j = index_of(x + 1u, y, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_SOLID || tj == CELL_TYPE_FLUID || tj == CELL_TYPE_SOURCE)
                {
                    sum += curr[j] - curr[i];
                }
            }

            if (y > 0u)
            {
                const uint j = index_of(x, y - 1u, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_SOLID || tj == CELL_TYPE_FLUID || tj == CELL_TYPE_SOURCE)
                {
                    sum += curr[j] - curr[i];
                }
            }

            if (y + 1u < height)
            {
                const uint j = index_of(x, y + 1u, width);
                const uchar tj = type[j];
                if (tj == CELL_TYPE_SOLID || tj == CELL_TYPE_FLUID || tj == CELL_TYPE_SOURCE)
                {
                    sum += curr[j] - curr[i];
                }
            }

            next[i] = curr[i] + sum * diffusion_dt;

            break;
        }

        default:
        {
            next[i] = curr[i];
            break;
        }
    }
}

__kernel void apply_temperature_boundaries
(
    __global const uchar* type,
    __global float* temperature,
    uint width,
    uint height,
    float min_temperature,
    float max_temperature
)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if (x >= width || y >= height)
    {
        return;
    }

    const uint i = index_of(x, y, width);
    const uchar ti = type[i];

    switch (ti)
    {
        case CELL_TYPE_SOLID:
        case CELL_TYPE_FLUID:
        {
            temperature[i] = clamp(temperature[i], min_temperature, max_temperature);
            break;
        }

        case CELL_TYPE_SOURCE:
        {
            temperature[i] = max_temperature;
            break;
        }

        default:
        {
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
    const float u = clamp((temp - min_temp) / (max_temp - min_temp), 0.0f, 1.0f);

    if (u < 0.25f)
    {
        const float t = u / 0.25f;
        return (uchar4)(0, lerp_uchar(0, 255, t), 255, 255);
    }

    if (u < 0.50f)
    {
        const float t = (u - 0.25f) / 0.25f;
        return (uchar4)(0, 255, lerp_uchar(255, 0, t), 255);
    }

    if (u < 0.75f)
    {
        const float t = (u - 0.50f) / 0.25f;
        return (uchar4)(lerp_uchar(0, 255, t), 255, 0, 255);
    }

    const float t = (u - 0.75f) / 0.25f;
    return (uchar4)(255, lerp_uchar(255, 0, t), 0, 255);
}

__kernel void render_heat
(
    __global const uchar* type,
    __global const float* temperature,
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

    const uint i = index_of(x, y, width);
    const uchar ti = type[i];

    switch (ti)
    {
        case CELL_TYPE_SOURCE:
        {
            pixels[i] = (uchar4)(255, 255, 255, 255);
            break;
        }

        case CELL_TYPE_VACUUM:
        {
            pixels[i] = (uchar4)(0, 0, 0, 255);
            break;
        }

        case CELL_TYPE_SOLID:
        case CELL_TYPE_FLUID:
        {
            pixels[i] = temperature_to_color(temperature[i], min_temp, max_temp);
            break;
        }
    }
}
