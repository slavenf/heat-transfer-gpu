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

layout(binding = 0)
uniform sampler2D curr_;

layout(binding = 1, r32f)
uniform writeonly image2D next_;

layout(binding = 2)
uniform usampler2D type_;

uniform float diffusion_;

uniform ivec2 size_;

#define CELL_TYPE_VACUUM    0u
#define CELL_TYPE_METAL     1u
#define CELL_TYPE_SOURCE    2u

uint cell_type(ivec2 p)
{
    return texelFetch(type_, p, 0).r;
}

float curr_temp(ivec2 p)
{
    return texelFetch(curr_, p, 0).r;
}

bool is_conductive(ivec2 p)
{
    const uint t = cell_type(p);
    return t == CELL_TYPE_METAL || t == CELL_TYPE_SOURCE;
}

void main()
{
    const ivec2 p = ivec2(gl_GlobalInvocationID.xy);

    if (p.x >= size_.x || p.y >= size_.y)
    {
        return;
    }

    switch (cell_type(p))
    {
        case CELL_TYPE_VACUUM:
        case CELL_TYPE_SOURCE:
        {
            imageStore(next_, p, vec4(curr_temp(p), 0.0, 0.0, 1.0));
            break;
        }

        case CELL_TYPE_METAL:
        {
            const float center = curr_temp(p);

            float sum = 0.0;

            if (p.x > 0)
            {
                const ivec2 j = ivec2(p.x - 1, p.y);
                if (is_conductive(j))
                {
                    sum += curr_temp(j) - center;
                }
            }

            if (p.x + 1 < size_.x)
            {
                const ivec2 j = ivec2(p.x + 1, p.y);
                if (is_conductive(j))
                {
                    sum += curr_temp(j) - center;
                }
            }

            if (p.y > 0)
            {
                const ivec2 j = ivec2(p.x, p.y - 1);
                if (is_conductive(j))
                {
                    sum += curr_temp(j) - center;
                }
            }

            if (p.y + 1 < size_.y)
            {
                const ivec2 j = ivec2(p.x, p.y + 1);
                if (is_conductive(j))
                {
                    sum += curr_temp(j) - center;
                }
            }

            imageStore(next_, p, vec4(center + diffusion_ * sum, 0.0, 0.0, 1.0));

            break;
        }
    }
}
