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

#include <glad/glad.h> // MUST BE FIRST

#include "GpuSolverOpenGl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "Mesh.hpp"
#include "Real.hpp"
#include "SolverParameters.hpp"

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

extern unsigned char src_opengl_compute_shader_glsl[];
extern unsigned int  src_opengl_compute_shader_glsl_len;

extern unsigned char src_opengl_fullscreen_vertex_shader_glsl[];
extern unsigned int  src_opengl_fullscreen_vertex_shader_glsl_len;

extern unsigned char src_opengl_fullscreen_fragment_shader_glsl[];
extern unsigned int  src_opengl_fullscreen_fragment_shader_glsl_len;

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

static std::string shader_info_log(GLuint shader)
{
    GLint log_length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);

    std::string log(static_cast<std::size_t>(std::max(0, log_length)), '\0');
    if (log_length > 0)
    {
        glGetShaderInfoLog(shader, log_length, nullptr, log.data());
    }

    return log;
}

static std::string program_info_log(GLuint program)
{
    GLint log_length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);

    std::string log(static_cast<std::size_t>(std::max(0, log_length)), '\0');
    if (log_length > 0)
    {
        glGetProgramInfoLog(program, log_length, nullptr, log.data());
    }

    return log;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

struct ShaderSource
{
    GLenum type;
    const unsigned char* source;
    unsigned int source_length;
};

static GLuint create_program(std::initializer_list<ShaderSource> shaders)
{
    std::vector<GLuint> compiled_shaders;
    compiled_shaders.reserve(shaders.size());

    for (const ShaderSource& shader_source : shaders)
    {
        const GLuint shader = glCreateShader(shader_source.type);
        const char* source = reinterpret_cast<const char*>(shader_source.source);
        const GLint source_length = static_cast<GLint>(shader_source.source_length);

        glShaderSource(shader, 1, &source, &source_length);
        glCompileShader(shader);

        GLint ok = GL_FALSE;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (ok != GL_TRUE)
        {
            const std::string log = shader_info_log(shader);

            glDeleteShader(shader);

            for (const GLuint compiled_shader : compiled_shaders)
            {
                glDeleteShader(compiled_shader);
            }

            throw std::runtime_error("OpenGL shader compilation failed:\n" + log);
        }

        compiled_shaders.push_back(shader);
    }

    const GLuint program = glCreateProgram();

    for (const GLuint shader : compiled_shaders)
    {
        glAttachShader(program, shader);
    }

    glLinkProgram(program);

    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (ok != GL_TRUE)
    {
        const std::string log = program_info_log(program);

        for (const GLuint shader : compiled_shaders)
        {
            glDetachShader(program, shader);
            glDeleteShader(shader);
        }

        glDeleteProgram(program);
        throw std::runtime_error("OpenGL program link failed:\n" + log);
    }

    for (const GLuint shader : compiled_shaders)
    {
        glDetachShader(program, shader);
        glDeleteShader(shader);
    }

    return program;
}

static void delete_program(GLuint& program)
{
    if (program != 0)
    {
        glDeleteProgram(program);
        program = 0;
    }
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

static GLuint create_texture_float(std::size_t width, std::size_t height)
{
    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D
    (
        GL_TEXTURE_2D,
        0,
        GL_R32F,
        static_cast<GLsizei>(width),
        static_cast<GLsizei>(height),
        0,
        GL_RED,
        GL_FLOAT,
        nullptr
    );
    glBindTexture(GL_TEXTURE_2D, 0);
    return texture;
}

static GLuint create_texture_vec2_float(std::size_t width, std::size_t height)
{
    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D
    (
        GL_TEXTURE_2D,
        0,
        GL_RG32F,
        static_cast<GLsizei>(width),
        static_cast<GLsizei>(height),
        0,
        GL_RG,
        GL_FLOAT,
        nullptr
    );
    glBindTexture(GL_TEXTURE_2D, 0);
    return texture;
}

static GLuint create_texture_uint8(std::size_t width, std::size_t height)
{
    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D
    (
        GL_TEXTURE_2D,
        0,
        GL_R8UI,
        static_cast<GLsizei>(width),
        static_cast<GLsizei>(height),
        0,
        GL_RED_INTEGER,
        GL_UNSIGNED_BYTE,
        nullptr
    );
    glBindTexture(GL_TEXTURE_2D, 0);
    return texture;
}

static void delete_texture(GLuint& texture)
{
    if (texture != 0)
    {
        glDeleteTextures(1, &texture);
        texture = 0;
    }
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

enum class ComputeKernel : GLint
{
    AddHeat = 0,
    AddBuoyancy = 1,
    ApplyVelocityBoundaries = 2,
    AdvectVelocity = 3,
    DiffuseVelocity = 4,
    ComputeDivergence = 5,
    ClearPressure = 6,
    SolvePressure = 7,
    SubtractPressureGradient = 8,
    AdvectTemperature = 9,
    DiffuseTemperature = 10,
    ApplyTemperatureBoundaries = 11
};

class GpuSolverOpenGl::Impl
{
public:

    Impl(const Mesh& mesh, const SolverParameters& parameters, sf::RenderWindow& window);

    ~Impl();

    void reset();

    void step(std::size_t num_iterations);

    void draw(sf::RenderTarget& target);

    void draw_velocity_field(sf::RenderTarget& target);

    Real average_temperature() const;

    Real max_displacement() const;

private:

    void add_heat(GLuint groups_x, GLuint groups_y);

    void add_buoyancy(GLuint groups_x, GLuint groups_y);

    void apply_velocity_boundaries(GLuint groups_x, GLuint groups_y);

    void advect_velocity(GLuint groups_x, GLuint groups_y);

    void diffuse_velocity(GLuint groups_x, GLuint groups_y);

    void project_velocity(GLuint groups_x, GLuint groups_y);

    void advect_temperature(GLuint groups_x, GLuint groups_y);

    void diffuse_temperature(GLuint groups_x, GLuint groups_y);

    void apply_temperature_boundaries(GLuint groups_x, GLuint groups_y);

    void run_kernel(ComputeKernel kernel, GLuint groups_x, GLuint groups_y);

    void render_to_texture(sf::RenderTarget& target);

private:

    const Mesh& mesh_;

    const SolverParameters& parameters_;

    sf::RenderTexture render_texture_;

    GLuint compute_program_ = 0;

    GLuint render_program_ = 0;

    GLuint tex_type_ = 0;
    GLuint tex_curr_temperature_ = 0;
    GLuint tex_next_temperature_ = 0;
    GLuint tex_curr_velocity_ = 0;
    GLuint tex_next_velocity_ = 0;
    GLuint tex_curr_pressure_ = 0;
    GLuint tex_next_pressure_ = 0;
    GLuint tex_divergence_ = 0;

    GLuint vao_ = 0;
};

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

GpuSolverOpenGl::Impl::Impl(const Mesh& mesh, const SolverParameters& parameters, sf::RenderWindow& window)
    : mesh_(mesh)
    , parameters_(parameters)
    , render_texture_(sf::Vector2u(mesh.width(), mesh.height()), window.getSettings())
{
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(sf::Context::getFunction)))
    {
        throw std::runtime_error("Failed to initialize GLAD");
    }

    if (!GLAD_GL_VERSION_4_3)
    {
        throw std::runtime_error("OpenGL 4.3 is required for GpuSolverOpenGl");
    }

    compute_program_ = create_program
    (
        {
            {GL_COMPUTE_SHADER, src_opengl_compute_shader_glsl, src_opengl_compute_shader_glsl_len}
        }
    );

    render_program_ = create_program
    (
        {
            {GL_VERTEX_SHADER, src_opengl_fullscreen_vertex_shader_glsl, src_opengl_fullscreen_vertex_shader_glsl_len},
            {GL_FRAGMENT_SHADER, src_opengl_fullscreen_fragment_shader_glsl, src_opengl_fullscreen_fragment_shader_glsl_len}
        }
    );

    tex_type_ = create_texture_uint8(mesh_.width(), mesh_.height());
    tex_curr_temperature_ = create_texture_float(mesh_.width(), mesh_.height());
    tex_next_temperature_ = create_texture_float(mesh_.width(), mesh_.height());
    tex_curr_velocity_ = create_texture_vec2_float(mesh_.width(), mesh_.height());
    tex_next_velocity_ = create_texture_vec2_float(mesh_.width(), mesh_.height());
    tex_curr_pressure_ = create_texture_float(mesh_.width(), mesh_.height());
    tex_next_pressure_ = create_texture_float(mesh_.width(), mesh_.height());
    tex_divergence_ = create_texture_float(mesh_.width(), mesh_.height());

    glGenVertexArrays(1, &vao_);

    glUseProgram(compute_program_);
    glUniform2i(glGetUniformLocation(compute_program_, "size_"), static_cast<GLint>(mesh_.width()), static_cast<GLint>(mesh_.height()));
    glUniform1f(glGetUniformLocation(compute_program_, "dt_"), parameters_.dt);
    glUniform1f(glGetUniformLocation(compute_program_, "min_temperature_"), mesh_.min_temperature());
    glUniform1f(glGetUniformLocation(compute_program_, "max_temperature_"), mesh_.max_temperature());
    glUniform1f(glGetUniformLocation(compute_program_, "source_heat_transfer_"), parameters_.source_heat_transfer);
    glUniform1f(glGetUniformLocation(compute_program_, "buoyancy_"), parameters_.buoyancy);
    glUniform1f(glGetUniformLocation(compute_program_, "velocity_damping_"), parameters_.velocity_damping);
    glUniform1f(glGetUniformLocation(compute_program_, "viscosity_dt_"), parameters_.viscosity * parameters_.dt);
    glUniform1f(glGetUniformLocation(compute_program_, "diffusion_dt_"), parameters_.thermal_diffusion * parameters_.dt);

    glUseProgram(render_program_);
    glUniform1i(glGetUniformLocation(render_program_, "curr_"), 0);
    glUniform1i(glGetUniformLocation(render_program_, "type_"), 1);
    glUniform1f(glGetUniformLocation(render_program_, "min_temp_"), mesh_.min_temperature());
    glUniform1f(glGetUniformLocation(render_program_, "max_temp_"), mesh_.max_temperature());
    glUniform2i(glGetUniformLocation(render_program_, "size_"), static_cast<GLint>(mesh_.width()), static_cast<GLint>(mesh_.height()));

    reset();
}

GpuSolverOpenGl::Impl::~Impl()
{
    if (vao_ != 0)
    {
        glDeleteVertexArrays(1, &vao_);
    }

    delete_texture(tex_divergence_);
    delete_texture(tex_next_pressure_);
    delete_texture(tex_curr_pressure_);
    delete_texture(tex_next_velocity_);
    delete_texture(tex_curr_velocity_);
    delete_texture(tex_next_temperature_);
    delete_texture(tex_curr_temperature_);
    delete_texture(tex_type_);

    delete_program(render_program_);
    delete_program(compute_program_);
}

void GpuSolverOpenGl::Impl::reset()
{
    const std::size_t cell_count = mesh_.width() * mesh_.height();

    std::vector<std::uint8_t> cell_type(cell_count);
    std::vector<float> initial_temperature(cell_count);

    for (std::size_t i = 0; i < cell_count; ++i)
    {
        cell_type[i] = static_cast<std::uint8_t>(mesh_.type(i));
        initial_temperature[i] = mesh_.initial_temperature(i);
    }

    std::vector<float> zero_scalar(cell_count, 0.0f);
    std::vector<std::array<float, 2>> zero_vector(cell_count, {0.0f, 0.0f});

    // OpenGL assumes by default that each uploaded row starts on a 4-byte boundary.
    // The cell type texture stores one byte per cell, so a row can be any number of
    // bytes wide. Use alignment 1 while uploading it so OpenGL does not expect
    // padding bytes at the end of rows whose width is not divisible by 4.
    GLint unpack_alignment = 0;
    // Save the current unpack alignment because this is global OpenGL state.
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &unpack_alignment);
    // Alignment 1 means rows are read as tightly packed bytes, with no padding.
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glBindTexture(GL_TEXTURE_2D, tex_type_);
    glTexSubImage2D
    (
        GL_TEXTURE_2D,
        0,
        0,
        0,
        static_cast<GLsizei>(mesh_.width()),
        static_cast<GLsizei>(mesh_.height()),
        GL_RED_INTEGER,
        GL_UNSIGNED_BYTE,
        cell_type.data()
    );

    // Restore the previous unpack alignment for later texture uploads.
    glPixelStorei(GL_UNPACK_ALIGNMENT, unpack_alignment);

    glBindTexture(GL_TEXTURE_2D, tex_curr_temperature_);
    glTexSubImage2D
    (
        GL_TEXTURE_2D,
        0,
        0,
        0,
        static_cast<GLsizei>(mesh_.width()),
        static_cast<GLsizei>(mesh_.height()),
        GL_RED,
        GL_FLOAT,
        initial_temperature.data()
    );

    glBindTexture(GL_TEXTURE_2D, tex_next_temperature_);
    glTexSubImage2D
    (
        GL_TEXTURE_2D,
        0,
        0,
        0,
        static_cast<GLsizei>(mesh_.width()),
        static_cast<GLsizei>(mesh_.height()),
        GL_RED,
        GL_FLOAT,
        initial_temperature.data()
    );

    glBindTexture(GL_TEXTURE_2D, tex_curr_velocity_);
    glTexSubImage2D
    (
        GL_TEXTURE_2D,
        0,
        0,
        0,
        static_cast<GLsizei>(mesh_.width()),
        static_cast<GLsizei>(mesh_.height()),
        GL_RG,
        GL_FLOAT,
        zero_vector.data()
    );

    glBindTexture(GL_TEXTURE_2D, tex_next_velocity_);
    glTexSubImage2D
    (
        GL_TEXTURE_2D,
        0,
        0,
        0,
        static_cast<GLsizei>(mesh_.width()),
        static_cast<GLsizei>(mesh_.height()),
        GL_RG,
        GL_FLOAT,
        zero_vector.data()
    );

    glBindTexture(GL_TEXTURE_2D, tex_curr_pressure_);
    glTexSubImage2D
    (
        GL_TEXTURE_2D,
        0,
        0,
        0,
        static_cast<GLsizei>(mesh_.width()),
        static_cast<GLsizei>(mesh_.height()),
        GL_RED,
        GL_FLOAT,
        zero_scalar.data()
    );

    glBindTexture(GL_TEXTURE_2D, tex_next_pressure_);
    glTexSubImage2D
    (
        GL_TEXTURE_2D,
        0,
        0,
        0,
        static_cast<GLsizei>(mesh_.width()),
        static_cast<GLsizei>(mesh_.height()),
        GL_RED,
        GL_FLOAT,
        zero_scalar.data()
    );

    glBindTexture(GL_TEXTURE_2D, tex_divergence_);
    glTexSubImage2D
    (
        GL_TEXTURE_2D,
        0,
        0,
        0,
        static_cast<GLsizei>(mesh_.width()),
        static_cast<GLsizei>(mesh_.height()),
        GL_RED,
        GL_FLOAT,
        zero_scalar.data()
    );

    glBindTexture(GL_TEXTURE_2D, 0);
}

void GpuSolverOpenGl::Impl::step(std::size_t num_iterations)
{
    const GLuint groups_x = static_cast<GLuint>((mesh_.width() + 15) / 16);
    const GLuint groups_y = static_cast<GLuint>((mesh_.height() + 15) / 16);

    while (num_iterations-- > 0)
    {
        add_heat(groups_x, groups_y);

        add_buoyancy(groups_x, groups_y);

        apply_velocity_boundaries(groups_x, groups_y);

        advect_velocity(groups_x, groups_y);

        apply_velocity_boundaries(groups_x, groups_y);

        diffuse_velocity(groups_x, groups_y);

        apply_velocity_boundaries(groups_x, groups_y);

        project_velocity(groups_x, groups_y);

        apply_velocity_boundaries(groups_x, groups_y);

        advect_temperature(groups_x, groups_y);

        diffuse_temperature(groups_x, groups_y);

        apply_temperature_boundaries(groups_x, groups_y);
    }

    glFinish();
}

void GpuSolverOpenGl::Impl::draw(sf::RenderTarget& target)
{
    render_to_texture(target);
    target.resetGLStates();
    target.draw(sf::Sprite(render_texture_.getTexture()));
}

void GpuSolverOpenGl::Impl::draw_velocity_field(sf::RenderTarget& target)
{
    const std::size_t cell_count = mesh_.width() * mesh_.height();
    std::vector<std::array<float, 2>> velocity(cell_count);

    glBindTexture(GL_TEXTURE_2D, tex_curr_velocity_);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_FLOAT, velocity.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    sf::VertexArray lines(sf::PrimitiveType::Lines);

    for (std::size_t y = 0; y < mesh_.height(); y += 2)
    {
        for (std::size_t x = 0; x < mesh_.width(); x += 2)
        {
            const auto i = mesh_.index(x, y);
            const auto ti = mesh_.type(i);

            if (ti == CellType::Fluid)
            {
                const float vx = velocity[i][0];
                const float vy = velocity[i][1];
                const float length = std::sqrt(vx * vx + vy * vy);

                if (length > 0.0f)
                {
                    const sf::Vector2f start(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
                    const sf::Vector2f end(start.x + vx / length, start.y + vy / length);

                    lines.append(sf::Vertex(start, sf::Color::White));
                    lines.append(sf::Vertex(end, sf::Color::White));
                }
            }
        }
    }

    target.draw(lines);
}

Real GpuSolverOpenGl::Impl::average_temperature() const
{
    const std::size_t cell_count = mesh_.width() * mesh_.height();
    std::vector<float> temperature(cell_count);

    glBindTexture(GL_TEXTURE_2D, tex_curr_temperature_);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, temperature.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    float sum = 0.0f;
    std::size_t count = 0;

    for (std::size_t i = 0; i < cell_count; ++i)
    {
        const auto ti = mesh_.type(i);

        if (ti == CellType::Fluid || ti == CellType::Solid)
        {
            sum += temperature[i];
            ++count;
        }
    }

    return static_cast<Real>(sum / count);
}

Real GpuSolverOpenGl::Impl::max_displacement() const
{
    const std::size_t cell_count = mesh_.width() * mesh_.height();
    std::vector<std::array<float, 2>> velocity(cell_count);

    glBindTexture(GL_TEXTURE_2D, tex_curr_velocity_);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_FLOAT, velocity.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    float max_speed = 0.0f;

    for (std::size_t i = 0; i < cell_count; ++i)
    {
        const auto ti = mesh_.type(i);

        if (ti == CellType::Fluid)
        {
            const float vx = velocity[i][0];
            const float vy = velocity[i][1];
            const float speed = std::sqrt(vx * vx + vy * vy);

            if (speed > max_speed)
            {
                max_speed = speed;
            }
        }
    }

    return static_cast<Real>(max_speed) * parameters_.dt;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void GpuSolverOpenGl::Impl::add_heat(GLuint groups_x, GLuint groups_y)
{
    run_kernel(ComputeKernel::AddHeat, groups_x, groups_y);
}

void GpuSolverOpenGl::Impl::add_buoyancy(GLuint groups_x, GLuint groups_y)
{
    run_kernel(ComputeKernel::AddBuoyancy, groups_x, groups_y);
}

void GpuSolverOpenGl::Impl::apply_velocity_boundaries(GLuint groups_x, GLuint groups_y)
{
    run_kernel(ComputeKernel::ApplyVelocityBoundaries, groups_x, groups_y);
}

void GpuSolverOpenGl::Impl::advect_velocity(GLuint groups_x, GLuint groups_y)
{
    run_kernel(ComputeKernel::AdvectVelocity, groups_x, groups_y);
    std::swap(tex_curr_velocity_, tex_next_velocity_);
}

void GpuSolverOpenGl::Impl::diffuse_velocity(GLuint groups_x, GLuint groups_y)
{
    run_kernel(ComputeKernel::DiffuseVelocity, groups_x, groups_y);
    std::swap(tex_curr_velocity_, tex_next_velocity_);
}

void GpuSolverOpenGl::Impl::project_velocity(GLuint groups_x, GLuint groups_y)
{
    run_kernel(ComputeKernel::ComputeDivergence, groups_x, groups_y);
    run_kernel(ComputeKernel::ClearPressure, groups_x, groups_y);

    for (int iter = 0; iter < parameters_.pressure_iterations; ++iter)
    {
        run_kernel(ComputeKernel::SolvePressure, groups_x, groups_y);
        std::swap(tex_curr_pressure_, tex_next_pressure_);
    }

    run_kernel(ComputeKernel::SubtractPressureGradient, groups_x, groups_y);
}

void GpuSolverOpenGl::Impl::advect_temperature(GLuint groups_x, GLuint groups_y)
{
    run_kernel(ComputeKernel::AdvectTemperature, groups_x, groups_y);
    std::swap(tex_curr_temperature_, tex_next_temperature_);
}

void GpuSolverOpenGl::Impl::diffuse_temperature(GLuint groups_x, GLuint groups_y)
{
    run_kernel(ComputeKernel::DiffuseTemperature, groups_x, groups_y);
    std::swap(tex_curr_temperature_, tex_next_temperature_);
}

void GpuSolverOpenGl::Impl::apply_temperature_boundaries(GLuint groups_x, GLuint groups_y)
{
    run_kernel(ComputeKernel::ApplyTemperatureBoundaries, groups_x, groups_y);
}

void GpuSolverOpenGl::Impl::run_kernel(ComputeKernel kernel, GLuint groups_x, GLuint groups_y)
{
    glUseProgram(compute_program_);

    glUniform1i(glGetUniformLocation(compute_program_, "kernel_"), static_cast<GLint>(kernel));

    glBindImageTexture(0, tex_type_, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8UI);
    glBindImageTexture(1, tex_curr_temperature_, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);
    glBindImageTexture(2, tex_next_temperature_, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);
    glBindImageTexture(3, tex_curr_velocity_, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RG32F);
    glBindImageTexture(4, tex_next_velocity_, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RG32F);
    glBindImageTexture(5, tex_curr_pressure_, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);
    glBindImageTexture(6, tex_next_pressure_, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);
    glBindImageTexture(7, tex_divergence_, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);

    glDispatchCompute(groups_x, groups_y, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT);
}

void GpuSolverOpenGl::Impl::render_to_texture(sf::RenderTarget& target)
{
    if (!render_texture_.setActive(true))
    {
        throw std::runtime_error("Failed to activate OpenGL render texture");
    }

    glViewport
    (
        0,
        0,
        static_cast<GLsizei>(mesh_.width()),
        static_cast<GLsizei>(mesh_.height())
    );

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(render_program_);
    glBindVertexArray(vao_);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_curr_temperature_);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_type_);

    glDrawArrays(GL_TRIANGLES, 0, 3);

    glBindVertexArray(0);

    render_texture_.display();

    if (!target.setActive(true))
    {
        throw std::runtime_error("Failed to reactivate OpenGL render target");
    }
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

GpuSolverOpenGl::GpuSolverOpenGl(const Mesh& mesh, const SolverParameters& parameters, sf::RenderWindow& window)
{
    #ifdef USE_DOUBLE
    (void)mesh;
    (void)parameters;
    (void)window;

    throw std::runtime_error("The OpenGL solver uses 32-bit float textures and does not support USE_DOUBLE.");
    #else
    impl_ = std::make_unique<Impl>(mesh, parameters, window);
    #endif
}

GpuSolverOpenGl::~GpuSolverOpenGl() = default;

void GpuSolverOpenGl::reset()
{
    impl_->reset();
}

void GpuSolverOpenGl::step(std::size_t num_iterations)
{
    impl_->step(num_iterations);
}

void GpuSolverOpenGl::draw(sf::RenderTarget& target)
{
    impl_->draw(target);
}

void GpuSolverOpenGl::draw_velocity_field(sf::RenderTarget& target)
{
    impl_->draw_velocity_field(target);
}

Real GpuSolverOpenGl::average_temperature() const
{
    return impl_->average_temperature();
}

Real GpuSolverOpenGl::max_displacement() const
{
    return impl_->max_displacement();
}
