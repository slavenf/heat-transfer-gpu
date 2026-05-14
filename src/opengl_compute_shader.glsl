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

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, r8ui)
uniform readonly uimage2D type_;

layout(binding = 1, r32f)
uniform image2D curr_temperature_;

layout(binding = 2, r32f)
uniform image2D next_temperature_;

layout(binding = 3, rg32f)
uniform image2D curr_velocity_;

layout(binding = 4, rg32f)
uniform image2D next_velocity_;

layout(binding = 5, r32f)
uniform image2D curr_pressure_;

layout(binding = 6, r32f)
uniform image2D next_pressure_;

layout(binding = 7, r32f)
uniform image2D divergence_;

uniform int kernel_;

uniform ivec2 size_;

uniform float dt_;
uniform float min_temperature_;
uniform float max_temperature_;
uniform float source_heat_transfer_;
uniform float buoyancy_;
uniform float velocity_damping_;
uniform float viscosity_dt_;
uniform float diffusion_dt_;

#define CELL_TYPE_SOLID     0u
#define CELL_TYPE_SOURCE    1u
#define CELL_TYPE_VACUUM    2u
#define CELL_TYPE_FLUID     3u

#define KERNEL_ADD_HEAT                     0
#define KERNEL_ADD_BUOYANCY                 1
#define KERNEL_APPLY_VELOCITY_BOUNDARIES    2
#define KERNEL_ADVECT_VELOCITY              3
#define KERNEL_DIFFUSE_VELOCITY             4
#define KERNEL_COMPUTE_DIVERGENCE           5
#define KERNEL_CLEAR_PRESSURE               6
#define KERNEL_SOLVE_PRESSURE               7
#define KERNEL_SUBTRACT_PRESSURE_GRADIENT   8
#define KERNEL_ADVECT_TEMPERATURE           9
#define KERNEL_DIFFUSE_TEMPERATURE          10
#define KERNEL_APPLY_TEMPERATURE_BOUNDARIES 11

uint cell_type(ivec2 p)
{
    return imageLoad(type_, p).r;
}

float curr_temperature(ivec2 p)
{
    return imageLoad(curr_temperature_, p).r;
}

void set_curr_temperature(ivec2 p, float value)
{
    imageStore(curr_temperature_, p, vec4(value, 0.0, 0.0, 0.0));
}

void set_next_temperature(ivec2 p, float value)
{
    imageStore(next_temperature_, p, vec4(value, 0.0, 0.0, 0.0));
}

float curr_pressure(ivec2 p)
{
    return imageLoad(curr_pressure_, p).r;
}

void set_curr_pressure(ivec2 p, float value)
{
    imageStore(curr_pressure_, p, vec4(value, 0.0, 0.0, 0.0));
}

void set_next_pressure(ivec2 p, float value)
{
    imageStore(next_pressure_, p, vec4(value, 0.0, 0.0, 0.0));
}

vec2 curr_velocity(ivec2 p)
{
    return imageLoad(curr_velocity_, p).xy;
}

void set_curr_velocity(ivec2 p, vec2 value)
{
    imageStore(curr_velocity_, p, vec4(value, 0.0, 0.0));
}

void set_next_velocity(ivec2 p, vec2 value)
{
    imageStore(next_velocity_, p, vec4(value, 0.0, 0.0));
}

float divergence_at(ivec2 p)
{
    return imageLoad(divergence_, p).r;
}

void set_divergence(ivec2 p, float value)
{
    imageStore(divergence_, p, vec4(value, 0.0, 0.0, 0.0));
}

float sample_temperature(float x, float y, float fallback)
{
    const int x0 = int(floor(x));
    const int y0 = int(floor(y));
    const int x1 = min(x0 + 1, size_.x - 1);
    const int y1 = min(y0 + 1, size_.y - 1);

    const float sx = x - float(x0);
    const float sy = y - float(y0);

    const ivec2 p00 = ivec2(x0, y0);
    const ivec2 p10 = ivec2(x1, y0);
    const ivec2 p01 = ivec2(x0, y1);
    const ivec2 p11 = ivec2(x1, y1);

    const float t00 = cell_type(p00) == CELL_TYPE_FLUID ? curr_temperature(p00) : fallback;
    const float t10 = cell_type(p10) == CELL_TYPE_FLUID ? curr_temperature(p10) : fallback;
    const float t01 = cell_type(p01) == CELL_TYPE_FLUID ? curr_temperature(p01) : fallback;
    const float t11 = cell_type(p11) == CELL_TYPE_FLUID ? curr_temperature(p11) : fallback;

    const float a = t00 + sx * (t10 - t00);
    const float b = t01 + sx * (t11 - t01);

    return a + sy * (b - a);
}

vec2 sample_velocity(float x, float y)
{
    const int x0 = int(floor(x));
    const int y0 = int(floor(y));
    const int x1 = min(x0 + 1, size_.x - 1);
    const int y1 = min(y0 + 1, size_.y - 1);

    const float sx = x - float(x0);
    const float sy = y - float(y0);

    const ivec2 p00 = ivec2(x0, y0);
    const ivec2 p10 = ivec2(x1, y0);
    const ivec2 p01 = ivec2(x0, y1);
    const ivec2 p11 = ivec2(x1, y1);

    const vec2 v00 = cell_type(p00) == CELL_TYPE_FLUID ? curr_velocity(p00) : vec2(0.0);
    const vec2 v10 = cell_type(p10) == CELL_TYPE_FLUID ? curr_velocity(p10) : vec2(0.0);
    const vec2 v01 = cell_type(p01) == CELL_TYPE_FLUID ? curr_velocity(p01) : vec2(0.0);
    const vec2 v11 = cell_type(p11) == CELL_TYPE_FLUID ? curr_velocity(p11) : vec2(0.0);

    const vec2 a = v00 + sx * (v10 - v00);
    const vec2 b = v01 + sx * (v11 - v01);

    return a + sy * (b - a);
}

void add_heat(ivec2 p)
{
    const uint ti = cell_type(p);

    switch (ti)
    {
        case CELL_TYPE_SOLID:
        case CELL_TYPE_FLUID:
        {
            bool touches_source = false;

            if (p.x > 0)
            {
                const ivec2 j = ivec2(p.x - 1, p.y);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_SOURCE)
                {
                    touches_source = true;
                }
            }

            if (p.x + 1 < size_.x)
            {
                const ivec2 j = ivec2(p.x + 1, p.y);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_SOURCE)
                {
                    touches_source = true;
                }
            }

            if (p.y > 0)
            {
                const ivec2 j = ivec2(p.x, p.y - 1);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_SOURCE)
                {
                    touches_source = true;
                }
            }

            if (p.y + 1 < size_.y)
            {
                const ivec2 j = ivec2(p.x, p.y + 1);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_SOURCE)
                {
                    touches_source = true;
                }
            }

            if (touches_source)
            {
                float temperature = curr_temperature(p) + source_heat_transfer_ * dt_;

                if (temperature > max_temperature_)
                {
                    temperature = max_temperature_;
                }

                set_curr_temperature(p, temperature);
            }

            break;
        }

        default:
        {
            break;
        }
    }
}

void add_buoyancy(ivec2 p)
{
    const uint ti = cell_type(p);

    switch (ti)
    {
        case CELL_TYPE_FLUID:
        {
            const vec2 velocity = curr_velocity(p) - vec2(0.0, buoyancy_ * (curr_temperature(p) - min_temperature_) * dt_);
            set_curr_velocity(p, velocity);
            break;
        }

        default:
        {
            break;
        }
    }
}

void apply_velocity_boundaries(ivec2 p)
{
    const uint ti = cell_type(p);

    switch (ti)
    {
        case CELL_TYPE_FLUID:
        {
            vec2 velocity = curr_velocity(p);

            if (p.x > 0)
            {
                const ivec2 j = ivec2(p.x - 1, p.y);
                const uint tj = cell_type(j);
                if (tj != CELL_TYPE_FLUID && velocity.x < 0.0)
                {
                    velocity.x = 0.0;
                }
            }
            else if (velocity.x < 0.0)
            {
                velocity.x = 0.0;
            }

            if (p.x + 1 < size_.x)
            {
                const ivec2 j = ivec2(p.x + 1, p.y);
                const uint tj = cell_type(j);
                if (tj != CELL_TYPE_FLUID && velocity.x > 0.0)
                {
                    velocity.x = 0.0;
                }
            }
            else if (velocity.x > 0.0)
            {
                velocity.x = 0.0;
            }

            if (p.y > 0)
            {
                const ivec2 j = ivec2(p.x, p.y - 1);
                const uint tj = cell_type(j);
                if (tj != CELL_TYPE_FLUID && velocity.y < 0.0)
                {
                    velocity.y = 0.0;
                }
            }
            else if (velocity.y < 0.0)
            {
                velocity.y = 0.0;
            }

            if (p.y + 1 < size_.y)
            {
                const ivec2 j = ivec2(p.x, p.y + 1);
                const uint tj = cell_type(j);
                if (tj != CELL_TYPE_FLUID && velocity.y > 0.0)
                {
                    velocity.y = 0.0;
                }
            }
            else if (velocity.y > 0.0)
            {
                velocity.y = 0.0;
            }

            set_curr_velocity(p, velocity);

            break;
        }

        default:
        {
            set_curr_velocity(p, vec2(0.0));
            break;
        }
    }
}

void advect_velocity(ivec2 p)
{
    const uint ti = cell_type(p);

    switch (ti)
    {
        case CELL_TYPE_FLUID:
        {
            const vec2 velocity = curr_velocity(p);
            const float old_x = clamp(float(p.x) - velocity.x * dt_, 0.0, float(size_.x - 1));
            const float old_y = clamp(float(p.y) - velocity.y * dt_, 0.0, float(size_.y - 1));

            set_next_velocity(p, sample_velocity(old_x, old_y) * velocity_damping_);

            break;
        }

        default:
        {
            set_next_velocity(p, vec2(0.0));
            break;
        }
    }
}

void diffuse_velocity(ivec2 p)
{
    const uint ti = cell_type(p);

    switch (ti)
    {
        case CELL_TYPE_FLUID:
        {
            const vec2 center = curr_velocity(p);
            vec2 sum = vec2(0.0);

            if (p.x > 0)
            {
                const ivec2 j = ivec2(p.x - 1, p.y);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_FLUID)
                {
                    sum += curr_velocity(j) - center;
                }
            }

            if (p.x + 1 < size_.x)
            {
                const ivec2 j = ivec2(p.x + 1, p.y);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_FLUID)
                {
                    sum += curr_velocity(j) - center;
                }
            }

            if (p.y > 0)
            {
                const ivec2 j = ivec2(p.x, p.y - 1);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_FLUID)
                {
                    sum += curr_velocity(j) - center;
                }
            }

            if (p.y + 1 < size_.y)
            {
                const ivec2 j = ivec2(p.x, p.y + 1);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_FLUID)
                {
                    sum += curr_velocity(j) - center;
                }
            }

            set_next_velocity(p, center + sum * viscosity_dt_);

            break;
        }

        default:
        {
            set_next_velocity(p, vec2(0.0));
            break;
        }
    }
}

void compute_divergence(ivec2 p)
{
    const uint ti = cell_type(p);

    switch (ti)
    {
        case CELL_TYPE_FLUID:
        {
            vec2 left_velocity = vec2(0.0);
            vec2 right_velocity = vec2(0.0);
            vec2 up_velocity = vec2(0.0);
            vec2 down_velocity = vec2(0.0);

            if (p.x > 0)
            {
                const ivec2 j = ivec2(p.x - 1, p.y);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_FLUID)
                {
                    left_velocity = curr_velocity(j);
                }
            }

            if (p.x + 1 < size_.x)
            {
                const ivec2 j = ivec2(p.x + 1, p.y);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_FLUID)
                {
                    right_velocity = curr_velocity(j);
                }
            }

            if (p.y > 0)
            {
                const ivec2 j = ivec2(p.x, p.y - 1);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_FLUID)
                {
                    up_velocity = curr_velocity(j);
                }
            }

            if (p.y + 1 < size_.y)
            {
                const ivec2 j = ivec2(p.x, p.y + 1);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_FLUID)
                {
                    down_velocity = curr_velocity(j);
                }
            }

            set_divergence(p, 0.5 * (right_velocity.x - left_velocity.x + down_velocity.y - up_velocity.y));

            break;
        }

        default:
        {
            set_divergence(p, 0.0);
            break;
        }
    }
}

void clear_pressure(ivec2 p)
{
    set_curr_pressure(p, 0.0);
    set_next_pressure(p, 0.0);
}

void solve_pressure(ivec2 p)
{
    const uint ti = cell_type(p);

    switch (ti)
    {
        case CELL_TYPE_FLUID:
        {
            const float center = curr_pressure(p);
            float left_pressure = center;
            float right_pressure = center;
            float up_pressure = center;
            float down_pressure = center;

            if (p.x > 0)
            {
                const ivec2 j = ivec2(p.x - 1, p.y);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_FLUID)
                {
                    left_pressure = curr_pressure(j);
                }
            }

            if (p.x + 1 < size_.x)
            {
                const ivec2 j = ivec2(p.x + 1, p.y);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_FLUID)
                {
                    right_pressure = curr_pressure(j);
                }
            }

            if (p.y > 0)
            {
                const ivec2 j = ivec2(p.x, p.y - 1);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_FLUID)
                {
                    up_pressure = curr_pressure(j);
                }
            }

            if (p.y + 1 < size_.y)
            {
                const ivec2 j = ivec2(p.x, p.y + 1);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_FLUID)
                {
                    down_pressure = curr_pressure(j);
                }
            }

            set_next_pressure(p, 0.25 * (left_pressure + right_pressure + up_pressure + down_pressure - divergence_at(p)));

            break;
        }

        default:
        {
            set_next_pressure(p, 0.0);
            break;
        }
    }
}

void subtract_pressure_gradient(ivec2 p)
{
    const uint ti = cell_type(p);

    switch (ti)
    {
        case CELL_TYPE_FLUID:
        {
            const float center = curr_pressure(p);
            float left_pressure = center;
            float right_pressure = center;
            float up_pressure = center;
            float down_pressure = center;

            if (p.x > 0)
            {
                const ivec2 j = ivec2(p.x - 1, p.y);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_FLUID)
                {
                    left_pressure = curr_pressure(j);
                }
            }

            if (p.x + 1 < size_.x)
            {
                const ivec2 j = ivec2(p.x + 1, p.y);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_FLUID)
                {
                    right_pressure = curr_pressure(j);
                }
            }

            if (p.y > 0)
            {
                const ivec2 j = ivec2(p.x, p.y - 1);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_FLUID)
                {
                    up_pressure = curr_pressure(j);
                }
            }

            if (p.y + 1 < size_.y)
            {
                const ivec2 j = ivec2(p.x, p.y + 1);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_FLUID)
                {
                    down_pressure = curr_pressure(j);
                }
            }

            vec2 velocity = curr_velocity(p);
            velocity.x -= 0.5 * (right_pressure - left_pressure);
            velocity.y -= 0.5 * (down_pressure - up_pressure);
            set_curr_velocity(p, velocity);

            break;
        }

        default:
        {
            break;
        }
    }
}

void advect_temperature(ivec2 p)
{
    const uint ti = cell_type(p);

    switch (ti)
    {
        case CELL_TYPE_FLUID:
        {
            const vec2 velocity = curr_velocity(p);
            const float old_x = clamp(float(p.x) - velocity.x * dt_, 0.0, float(size_.x - 1));
            const float old_y = clamp(float(p.y) - velocity.y * dt_, 0.0, float(size_.y - 1));

            set_next_temperature(p, sample_temperature(old_x, old_y, curr_temperature(p)));

            break;
        }

        default:
        {
            set_next_temperature(p, curr_temperature(p));
            break;
        }
    }
}

void diffuse_temperature(ivec2 p)
{
    const uint ti = cell_type(p);

    switch (ti)
    {
        case CELL_TYPE_SOLID:
        case CELL_TYPE_FLUID:
        {
            const float center = curr_temperature(p);
            float sum = 0.0;

            if (p.x > 0)
            {
                const ivec2 j = ivec2(p.x - 1, p.y);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_SOLID || tj == CELL_TYPE_FLUID || tj == CELL_TYPE_SOURCE)
                {
                    sum += curr_temperature(j) - center;
                }
            }

            if (p.x + 1 < size_.x)
            {
                const ivec2 j = ivec2(p.x + 1, p.y);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_SOLID || tj == CELL_TYPE_FLUID || tj == CELL_TYPE_SOURCE)
                {
                    sum += curr_temperature(j) - center;
                }
            }

            if (p.y > 0)
            {
                const ivec2 j = ivec2(p.x, p.y - 1);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_SOLID || tj == CELL_TYPE_FLUID || tj == CELL_TYPE_SOURCE)
                {
                    sum += curr_temperature(j) - center;
                }
            }

            if (p.y + 1 < size_.y)
            {
                const ivec2 j = ivec2(p.x, p.y + 1);
                const uint tj = cell_type(j);
                if (tj == CELL_TYPE_SOLID || tj == CELL_TYPE_FLUID || tj == CELL_TYPE_SOURCE)
                {
                    sum += curr_temperature(j) - center;
                }
            }

            set_next_temperature(p, center + sum * diffusion_dt_);

            break;
        }

        default:
        {
            set_next_temperature(p, curr_temperature(p));
            break;
        }
    }
}

void apply_temperature_boundaries(ivec2 p)
{
    const uint ti = cell_type(p);

    switch (ti)
    {
        case CELL_TYPE_SOLID:
        case CELL_TYPE_FLUID:
        {
            set_curr_temperature(p, clamp(curr_temperature(p), min_temperature_, max_temperature_));
            break;
        }

        case CELL_TYPE_SOURCE:
        {
            set_curr_temperature(p, max_temperature_);
            break;
        }

        default:
        {
            break;
        }
    }
}

void main()
{
    const ivec2 p = ivec2(gl_GlobalInvocationID.xy);

    if (p.x >= size_.x || p.y >= size_.y)
    {
        return;
    }

    switch (kernel_)
    {
        case KERNEL_ADD_HEAT:
            add_heat(p);
            break;

        case KERNEL_ADD_BUOYANCY:
            add_buoyancy(p);
            break;

        case KERNEL_APPLY_VELOCITY_BOUNDARIES:
            apply_velocity_boundaries(p);
            break;

        case KERNEL_ADVECT_VELOCITY:
            advect_velocity(p);
            break;

        case KERNEL_DIFFUSE_VELOCITY:
            diffuse_velocity(p);
            break;

        case KERNEL_COMPUTE_DIVERGENCE:
            compute_divergence(p);
            break;

        case KERNEL_CLEAR_PRESSURE:
            clear_pressure(p);
            break;

        case KERNEL_SOLVE_PRESSURE:
            solve_pressure(p);
            break;

        case KERNEL_SUBTRACT_PRESSURE_GRADIENT:
            subtract_pressure_gradient(p);
            break;

        case KERNEL_ADVECT_TEMPERATURE:
            advect_temperature(p);
            break;

        case KERNEL_DIFFUSE_TEMPERATURE:
            diffuse_temperature(p);
            break;

        case KERNEL_APPLY_TEMPERATURE_BOUNDARIES:
            apply_temperature_boundaries(p);
            break;
    }
}
