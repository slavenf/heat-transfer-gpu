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

#ifdef USE_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double Real;
typedef double2 Real2;
#else
typedef float Real;
typedef float2 Real2;
#endif

#define REAL(x) ((Real)(x))
#define REAL2(x, y) ((Real2)(REAL(x), REAL(y)))

#define CELL_TYPE_SOLID     0u
#define CELL_TYPE_SOURCE    1u
#define CELL_TYPE_VACUUM    2u
#define CELL_TYPE_FLUID     3u

uint index_of(uint x, uint y, uint width)
{
    return y * width + x;
}

Real sample_temperature
(
    __global const uchar* type,
    __global const Real* temperature,
    Real x,
    Real y,
    uint width,
    uint height,
    Real fallback
)
{
    const uint x0 = (uint)floor(x);
    const uint y0 = (uint)floor(y);
    const uint x1 = min(x0 + 1u, width - 1u);
    const uint y1 = min(y0 + 1u, height - 1u);

    const Real sx = x - REAL(x0);
    const Real sy = y - REAL(y0);

    const uint i00 = index_of(x0, y0, width);
    const uint i10 = index_of(x1, y0, width);
    const uint i01 = index_of(x0, y1, width);
    const uint i11 = index_of(x1, y1, width);

    const Real t00 = type[i00] == CELL_TYPE_FLUID ? temperature[i00] : fallback;
    const Real t10 = type[i10] == CELL_TYPE_FLUID ? temperature[i10] : fallback;
    const Real t01 = type[i01] == CELL_TYPE_FLUID ? temperature[i01] : fallback;
    const Real t11 = type[i11] == CELL_TYPE_FLUID ? temperature[i11] : fallback;

    const Real a = t00 + sx * (t10 - t00);
    const Real b = t01 + sx * (t11 - t01);

    return a + sy * (b - a);
}

Real2 sample_velocity
(
    __global const uchar* type,
    __global const Real2* velocity,
    Real x,
    Real y,
    uint width,
    uint height
)
{
    const uint x0 = (uint)floor(x);
    const uint y0 = (uint)floor(y);
    const uint x1 = min(x0 + 1u, width - 1u);
    const uint y1 = min(y0 + 1u, height - 1u);

    const Real sx = x - REAL(x0);
    const Real sy = y - REAL(y0);

    const uint i00 = index_of(x0, y0, width);
    const uint i10 = index_of(x1, y0, width);
    const uint i01 = index_of(x0, y1, width);
    const uint i11 = index_of(x1, y1, width);

    const Real2 v00 = type[i00] == CELL_TYPE_FLUID ? velocity[i00] : REAL2(0.0, 0.0);
    const Real2 v10 = type[i10] == CELL_TYPE_FLUID ? velocity[i10] : REAL2(0.0, 0.0);
    const Real2 v01 = type[i01] == CELL_TYPE_FLUID ? velocity[i01] : REAL2(0.0, 0.0);
    const Real2 v11 = type[i11] == CELL_TYPE_FLUID ? velocity[i11] : REAL2(0.0, 0.0);

    const Real2 a = v00 + sx * (v10 - v00);
    const Real2 b = v01 + sx * (v11 - v01);

    return a + sy * (b - a);
}

__kernel void add_heat
(
    __global const uchar* type,
    __global Real* temperature,
    uint width,
    uint height,
    Real source_heat_transfer,
    Real dt,
    Real max_temperature
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
    __global const Real* temperature,
    __global Real2* velocity,
    uint width,
    uint height,
    Real ambient_temperature,
    Real buoyancy,
    Real dt
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
    __global Real2* velocity,
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
                if (tj != CELL_TYPE_FLUID && velocity[i].x < REAL(0.0))
                {
                    velocity[i].x = REAL(0.0);
                }
            }
            else if (velocity[i].x < REAL(0.0))
            {
                velocity[i].x = REAL(0.0);
            }

            if (x + 1u < width)
            {
                const uint j = index_of(x + 1u, y, width);
                const uchar tj = type[j];
                if (tj != CELL_TYPE_FLUID && velocity[i].x > REAL(0.0))
                {
                    velocity[i].x = REAL(0.0);
                }
            }
            else if (velocity[i].x > REAL(0.0))
            {
                velocity[i].x = REAL(0.0);
            }

            if (y > 0u)
            {
                const uint j = index_of(x, y - 1u, width);
                const uchar tj = type[j];
                if (tj != CELL_TYPE_FLUID && velocity[i].y < REAL(0.0))
                {
                    velocity[i].y = REAL(0.0);
                }
            }
            else if (velocity[i].y < REAL(0.0))
            {
                velocity[i].y = REAL(0.0);
            }

            if (y + 1u < height)
            {
                const uint j = index_of(x, y + 1u, width);
                const uchar tj = type[j];
                if (tj != CELL_TYPE_FLUID && velocity[i].y > REAL(0.0))
                {
                    velocity[i].y = REAL(0.0);
                }
            }
            else if (velocity[i].y > REAL(0.0))
            {
                velocity[i].y = REAL(0.0);
            }

            break;
        }

        default:
        {
            velocity[i] = REAL2(0.0, 0.0);
            break;
        }
    }
}

__kernel void advect_velocity
(
    __global const uchar* type,
    __global const Real2* curr,
    __global Real2* next,
    uint width,
    uint height,
    Real dt,
    Real damping
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
            const Real old_x = clamp(REAL(x) - curr[i].x * dt, REAL(0.0), REAL(width - 1u));
            const Real old_y = clamp(REAL(y) - curr[i].y * dt, REAL(0.0), REAL(height - 1u));

            next[i] = sample_velocity(type, curr, old_x, old_y, width, height);
            next[i] *= damping;

            break;
        }

        default:
        {
            next[i] = REAL2(0.0, 0.0);
            break;
        }
    }
}

__kernel void diffuse_velocity
(
    __global const uchar* type,
    __global const Real2* curr,
    __global Real2* next,
    uint width,
    uint height,
    Real viscosity_dt
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
            Real2 sum = REAL2(0.0, 0.0);

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
            next[i] = REAL2(0.0, 0.0);
            break;
        }
    }
}

__kernel void compute_divergence
(
    __global const uchar* type,
    __global const Real2* velocity,
    __global Real* divergence,
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
            Real2 left_velocity = REAL2(0.0, 0.0);
            Real2 right_velocity = REAL2(0.0, 0.0);
            Real2 up_velocity = REAL2(0.0, 0.0);
            Real2 down_velocity = REAL2(0.0, 0.0);

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

            divergence[i] = REAL(0.5) * (right_velocity.x - left_velocity.x + down_velocity.y - up_velocity.y);

            break;
        }

        default:
        {
            divergence[i] = REAL(0.0);
            break;
        }
    }
}

__kernel void clear_pressure
(
    __global Real* curr,
    __global Real* next,
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
    curr[i] = REAL(0.0);
    next[i] = REAL(0.0);
}

__kernel void solve_pressure
(
    __global const uchar* type,
    __global const Real* curr,
    __global Real* next,
    __global const Real* divergence,
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
            Real left_pressure = curr[i];
            Real right_pressure = curr[i];
            Real up_pressure = curr[i];
            Real down_pressure = curr[i];

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

            next[i] = REAL(0.25) * (left_pressure + right_pressure + up_pressure + down_pressure - divergence[i]);

            break;
        }

        default:
        {
            next[i] = REAL(0.0);
            break;
        }
    }
}

__kernel void subtract_pressure_gradient
(
    __global const uchar* type,
    __global const Real* pressure,
    __global Real2* velocity,
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
            Real left_pressure = pressure[i];
            Real right_pressure = pressure[i];
            Real up_pressure = pressure[i];
            Real down_pressure = pressure[i];

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

            velocity[i].x -= REAL(0.5) * (right_pressure - left_pressure);
            velocity[i].y -= REAL(0.5) * (down_pressure - up_pressure);

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
    __global const Real* curr,
    __global Real* next,
    __global const Real2* velocity,
    uint width,
    uint height,
    Real dt
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
            const Real old_x = clamp(REAL(x) - velocity[i].x * dt, REAL(0.0), REAL(width - 1u));
            const Real old_y = clamp(REAL(y) - velocity[i].y * dt, REAL(0.0), REAL(height - 1u));

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
    __global const Real* curr,
    __global Real* next,
    uint width,
    uint height,
    Real diffusion_dt
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
            Real sum = REAL(0.0);

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
    __global Real* temperature,
    uint width,
    uint height,
    Real min_temperature,
    Real max_temperature
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

uchar lerp_uchar(uchar a, uchar b, Real t)
{
    return convert_uchar_sat(REAL(a) + (REAL(b) - REAL(a)) * t);
}

uchar4 temperature_to_color(Real temp, Real min_temp, Real max_temp)
{
    const Real u = clamp((temp - min_temp) / (max_temp - min_temp), REAL(0.0), REAL(1.0));

    if (u < REAL(0.25))
    {
        const Real t = u / REAL(0.25);
        return (uchar4)(0, lerp_uchar(0, 255, t), 255, 255);
    }

    if (u < REAL(0.50))
    {
        const Real t = (u - REAL(0.25)) / REAL(0.25);
        return (uchar4)(0, 255, lerp_uchar(255, 0, t), 255);
    }

    if (u < REAL(0.75))
    {
        const Real t = (u - REAL(0.50)) / REAL(0.25);
        return (uchar4)(lerp_uchar(0, 255, t), 255, 0, 255);
    }

    const Real t = (u - REAL(0.75)) / REAL(0.25);
    return (uchar4)(255, lerp_uchar(255, 0, t), 0, 255);
}

__kernel void render_heat
(
    __global const uchar* type,
    __global const Real* temperature,
    __global uchar4* pixels,
    uint width,
    uint height,
    Real min_temp,
    Real max_temp
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
