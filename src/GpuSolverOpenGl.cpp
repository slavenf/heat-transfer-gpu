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
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

static GLuint compile_shader(GLenum type, const char* source, GLint source_length)
{
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, &source_length);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (ok == GL_TRUE)
    {
        return shader;
    }

    GLint log_length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);

    std::string log(static_cast<std::size_t>(std::max(0, log_length)), '\0');
    if (log_length > 0)
    {
        glGetShaderInfoLog(shader, log_length, nullptr, log.data());
    }

    glDeleteShader(shader);
    throw std::runtime_error("OpenGL shader compilation failed:\n" + log);
}

static GLuint link_program(std::initializer_list<GLuint> shaders)
{
    const GLuint program = glCreateProgram();

    for (const GLuint shader : shaders)
    {
        glAttachShader(program, shader);
    }

    glLinkProgram(program);

    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (ok == GL_TRUE)
    {
        for (const GLuint shader : shaders)
        {
            glDetachShader(program, shader);
            glDeleteShader(shader);
        }
        return program;
    }

    GLint log_length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);

    std::string log(static_cast<std::size_t>(std::max(0, log_length)), '\0');
    if (log_length > 0)
    {
        glGetProgramInfoLog(program, log_length, nullptr, log.data());
    }

    for (const GLuint shader : shaders)
    {
        glDetachShader(program, shader);
        glDeleteShader(shader);
    }

    glDeleteProgram(program);
    throw std::runtime_error("OpenGL program link failed:\n" + log);
}

static GLuint create_texture_float(std::size_t width, std::size_t height, const float* data)
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
        data
    );
    glBindTexture(GL_TEXTURE_2D, 0);
    return texture;
}

static GLuint create_texture_uint8(std::size_t width, std::size_t height, const std::uint8_t* data)
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
        data
    );
    glBindTexture(GL_TEXTURE_2D, 0);
    return texture;
}

static std::vector<float> create_initial_temperature(const Mesh& mesh)
{
    std::vector<float> initial_temperature(mesh.width() * mesh.height(), 0.0f);

    #pragma omp parallel for schedule(static)
    for (std::size_t y = 0; y < mesh.height(); ++y)
    {
        for (std::size_t x = 0; x < mesh.width(); ++x)
        {
            const std::size_t i = mesh.index(x, y);

            switch (mesh.type(i))
            {
                case CellType::Vacuum:
                {
                    // Vacuum has no inital temeprature
                    break;
                }

                case CellType::Source:
                {
                    initial_temperature[i] = 100.0f;
                    break;
                }

                case CellType::Metal:
                {
                    initial_temperature[i] = 0.0f;
                    break;
                }
            }
        }
    }

    return initial_temperature;
}

static std::vector<std::uint8_t> create_cell_type(const Mesh& mesh)
{
    std::vector<std::uint8_t> cell_type(mesh.width() * mesh.height(), 0);

    #pragma omp parallel for schedule(static)
    for (std::size_t y = 0; y < mesh.height(); ++y)
    {
        for (std::size_t x = 0; x < mesh.width(); ++x)
        {
            const std::size_t i = mesh.index(x, y);
            cell_type[i] = static_cast<std::uint8_t>(mesh.type(i));
        }
    }

    return cell_type;
}

///////////////////////////////////////////////////////////////////////////////

struct GpuSolverOpenGl::Impl
{
    GLuint compute_program = 0;
    GLuint render_program = 0;

    GLuint tex_type = 0;
    GLuint tex_curr = 0;
    GLuint tex_next = 0;

    GLuint vao = 0;

    ~Impl()
    {
        if (vao != 0)
        {
            glDeleteVertexArrays(1, &vao);
        }

        if (tex_type != 0)
        {
            glDeleteTextures(1, &tex_type);
        }

        if (tex_curr != 0)
        {
            glDeleteTextures(1, &tex_curr);
        }

        if (tex_next != 0)
        {
            glDeleteTextures(1, &tex_next);
        }

        if (render_program != 0)
        {
            glDeleteProgram(render_program);
        }

        if (compute_program != 0)
        {
            glDeleteProgram(compute_program);
        }
    }
};

GpuSolverOpenGl::GpuSolverOpenGl(const Mesh& mesh, sf::RenderWindow& window)
    : mesh_(mesh)
    , window_(window)
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

    // Create Impl after GLAD initialization
    impl_ = std::make_unique<Impl>();

    const std::vector<std::uint8_t> cell_type = create_cell_type(mesh_);
    const std::vector<float> initial_temperature = create_initial_temperature(mesh_);

    extern unsigned char src_opengl_compute_shader_glsl[];
    extern unsigned int  src_opengl_compute_shader_glsl_len;
    extern unsigned char src_opengl_fullscreen_vertex_shader_glsl[];
    extern unsigned int  src_opengl_fullscreen_vertex_shader_glsl_len;
    extern unsigned char src_opengl_fullscreen_fragment_shader_glsl[];
    extern unsigned int  src_opengl_fullscreen_fragment_shader_glsl_len;

    impl_->compute_program = link_program
    (
        {
            compile_shader(GL_COMPUTE_SHADER, reinterpret_cast<const char*>(src_opengl_compute_shader_glsl), src_opengl_compute_shader_glsl_len)
        }
    );

    impl_->render_program = link_program
    (
        {
            compile_shader(GL_VERTEX_SHADER, reinterpret_cast<const char*>(src_opengl_fullscreen_vertex_shader_glsl), src_opengl_fullscreen_vertex_shader_glsl_len),
            compile_shader(GL_FRAGMENT_SHADER, reinterpret_cast<const char*>(src_opengl_fullscreen_fragment_shader_glsl), src_opengl_fullscreen_fragment_shader_glsl_len)
        }
    );

    impl_->tex_type = create_texture_uint8(mesh_.width(), mesh_.height(), cell_type.data());
    impl_->tex_curr = create_texture_float(mesh_.width(), mesh_.height(), initial_temperature.data());
    impl_->tex_next = create_texture_float(mesh_.width(), mesh_.height(), initial_temperature.data());

    glGenVertexArrays(1, &impl_->vao);

    glUseProgram(impl_->compute_program);
    glUniform1i(glGetUniformLocation(impl_->compute_program, "curr_"), 0);
    glUniform1i(glGetUniformLocation(impl_->compute_program, "type_"), 2);
    glUniform1f(glGetUniformLocation(impl_->compute_program, "diffusion_"), diffusion_);
    glUniform2i(glGetUniformLocation(impl_->compute_program, "size_"), static_cast<GLint>(mesh_.width()), static_cast<GLint>(mesh_.height()));

    glUseProgram(impl_->render_program);
    glUniform1i(glGetUniformLocation(impl_->render_program, "curr_"), 0);
    glUniform1i(glGetUniformLocation(impl_->render_program, "type_"), 1);
    glUniform1f(glGetUniformLocation(impl_->render_program, "min_temp_"), 0.0f);
    glUniform1f(glGetUniformLocation(impl_->render_program, "max_temp_"), 100.0f);
    glUniform2i(glGetUniformLocation(impl_->render_program, "size_"), static_cast<GLint>(mesh_.width()), static_cast<GLint>(mesh_.height()));
}

GpuSolverOpenGl::~GpuSolverOpenGl() = default;

void GpuSolverOpenGl::reset()
{
    const std::vector<float> initial_temperature = create_initial_temperature(mesh_);

    glBindTexture(GL_TEXTURE_2D, impl_->tex_curr);
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

    glBindTexture(GL_TEXTURE_2D, 0);
}

void GpuSolverOpenGl::step(std::size_t num_iterations)
{
    const GLuint groups_x = static_cast<GLuint>((mesh_.width() + 15) / 16);
    const GLuint groups_y = static_cast<GLuint>((mesh_.height() + 15) / 16);

    glUseProgram(impl_->compute_program);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, impl_->tex_type);

    glActiveTexture(GL_TEXTURE0);

    while (num_iterations-- > 0)
    {
        glBindTexture(GL_TEXTURE_2D, impl_->tex_curr);

        glBindImageTexture
        (
            1,
            impl_->tex_next,
            0,
            GL_FALSE,
            0,
            GL_WRITE_ONLY,
            GL_R32F
        );

        glDispatchCompute(groups_x, groups_y, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

        std::swap(impl_->tex_curr, impl_->tex_next);
    }

    glFinish();
}

void GpuSolverOpenGl::render_to_texture()
{
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

    glUseProgram(impl_->render_program);
    glBindVertexArray(impl_->vao);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, impl_->tex_curr);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, impl_->tex_type);

    glDrawArrays(GL_TRIANGLES, 0, 3);

    glBindVertexArray(0);

    render_texture_.display();
}

void GpuSolverOpenGl::draw(sf::RenderTarget& target)
{
    render_to_texture();
    target.resetGLStates();
    target.draw(sf::Sprite(render_texture_.getTexture()));
}
