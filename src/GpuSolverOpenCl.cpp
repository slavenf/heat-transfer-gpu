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

#include "GpuSolverOpenCl.hpp"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Mesh.hpp"
#include "SolverParameters.hpp"

///////////////////////////////////////////////////////////////////////////////
// Declarations
///////////////////////////////////////////////////////////////////////////////

extern unsigned char src_opencl_solver_cl[];
extern unsigned int src_opencl_solver_cl_len;

///////////////////////////////////////////////////////////////////////////////
// OpenCL error handling functions
///////////////////////////////////////////////////////////////////////////////

static std::string opencl_error_string(cl_int error)
{
    return "OpenCL error " + std::to_string(error);
}

static void check_cl(cl_int error, const char* what)
{
    if (error != CL_SUCCESS)
    {
        throw std::runtime_error(std::string(what) + " failed: " + opencl_error_string(error));
    }
}

///////////////////////////////////////////////////////////////////////////////
// OpenCL device functions
///////////////////////////////////////////////////////////////////////////////

static std::string get_device_string(cl_device_id device, cl_device_info info)
{
    std::size_t size = 0;
    check_cl(clGetDeviceInfo(device, info, 0, nullptr, &size), "clGetDeviceInfo");

    std::string value(size, '\0');
    check_cl(clGetDeviceInfo(device, info, size, value.data(), nullptr), "clGetDeviceInfo");

    if (!value.empty() && value.back() == '\0')
    {
        value.pop_back();
    }

    return value;
}

static std::vector<cl_platform_id> get_platforms()
{
    cl_uint count = 0;
    check_cl(clGetPlatformIDs(0, nullptr, &count), "clGetPlatformIDs");

    std::vector<cl_platform_id> platforms(count);
    check_cl(clGetPlatformIDs(count, platforms.data(), nullptr), "clGetPlatformIDs");

    return platforms;
}

static std::vector<cl_device_id> get_devices(cl_platform_id platform, cl_device_type type)
{
    cl_uint count = 0;
    const cl_int count_result = clGetDeviceIDs(platform, type, 0, nullptr, &count);
    if (count_result == CL_DEVICE_NOT_FOUND)
    {
        return {};
    }
    check_cl(count_result, "clGetDeviceIDs");

    std::vector<cl_device_id> devices(count);
    check_cl(clGetDeviceIDs(platform, type, count, devices.data(), nullptr), "clGetDeviceIDs");

    return devices;
}

static cl_device_id choose_device()
{
    const std::vector<cl_platform_id> platforms = get_platforms();

    for (const cl_platform_id platform : platforms)
    {
        const std::vector<cl_device_id> devices = get_devices(platform, CL_DEVICE_TYPE_GPU);
        if (!devices.empty())
        {
            return devices.front();
        }
    }

    for (const cl_platform_id platform : platforms)
    {
        const std::vector<cl_device_id> devices = get_devices(platform, CL_DEVICE_TYPE_ALL);
        if (!devices.empty())
        {
            return devices.front();
        }
    }

    throw std::runtime_error("No OpenCL devices found");
}

///////////////////////////////////////////////////////////////////////////////
// OpenCL context functions
///////////////////////////////////////////////////////////////////////////////

static cl_context create_context(cl_device_id device)
{
    cl_int error = CL_SUCCESS;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
    check_cl(error, "clCreateContext");
    return context;
}

static void release_context(cl_context& context)
{
    if (context != nullptr)
    {
        clReleaseContext(context);
        context = nullptr;
    }
}

///////////////////////////////////////////////////////////////////////////////
// OpenCL command queue functions
///////////////////////////////////////////////////////////////////////////////

static cl_command_queue create_command_queue(cl_device_id device, cl_context context)
{
    cl_int error = CL_SUCCESS;
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);
    check_cl(error, "clCreateCommandQueue");
    return queue;
}

static void release_command_queue(cl_command_queue& queue)
{
    if (queue != nullptr)
    {
        clReleaseCommandQueue(queue);
        queue = nullptr;
    }
}

///////////////////////////////////////////////////////////////////////////////
// OpenCL program functions
///////////////////////////////////////////////////////////////////////////////

static cl_program create_and_build_program(cl_device_id device, cl_context context, const char* source, std::size_t source_length)
{
    cl_int error = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(context, 1, &source, &source_length, &error);
    check_cl(error, "clCreateProgramWithSource");

    error = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
    if (error == CL_SUCCESS)
    {
        return program;
    }

    std::size_t log_size = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

    std::string log(log_size, '\0');
    if (log_size > 0)
    {
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
    }

    clReleaseProgram(program);
    throw std::runtime_error("OpenCL program build failed:\n" + log);
}

static void release_program(cl_program& program)
{
    if (program != nullptr)
    {
        clReleaseProgram(program);
        program = nullptr;
    }
}

///////////////////////////////////////////////////////////////////////////////
// OpenCL kernel functions
///////////////////////////////////////////////////////////////////////////////

static cl_kernel create_kernel(cl_program program, const char* name)
{
    cl_int error = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program, name, &error);
    check_cl(error, "clCreateKernel");
    return kernel;
}

static void release_kernel(cl_kernel& kernel)
{
    if (kernel != nullptr)
    {
        clReleaseKernel(kernel);
        kernel = nullptr;
    }
}

template <typename T>
static void set_kernel_arg(cl_kernel kernel, cl_uint index, const T& value)
{
    check_cl(clSetKernelArg(kernel, index, sizeof(T), &value), "clSetKernelArg");
}

static void run_kernel(cl_command_queue command_queue, cl_kernel kernel, const std::size_t (&global)[2], const std::size_t (&local)[2])
{
    check_cl
    (
        clEnqueueNDRangeKernel
        (
            command_queue,
            kernel,
            2,
            nullptr,
            global,
            local,
            0,
            nullptr,
            nullptr
        ),
        "clEnqueueNDRangeKernel"
    );
}

///////////////////////////////////////////////////////////////////////////////
// OpenCL buffer functions
///////////////////////////////////////////////////////////////////////////////

static cl_mem create_buffer(const cl_context& context, cl_mem_flags flags, std::size_t size)
{
    cl_int error = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(context, flags, size, nullptr, &error);
    check_cl(error, "clCreateBuffer");
    return buffer;
}

static void release_mem_object(cl_mem& object)
{
    if (object != nullptr)
    {
        clReleaseMemObject(object);
        object = nullptr;
    }
}

///////////////////////////////////////////////////////////////////////////////
// Helper functions
///////////////////////////////////////////////////////////////////////////////

static std::size_t round_up(std::size_t value, std::size_t multiple)
{
    return ((value + multiple - 1) / multiple) * multiple;
}

///////////////////////////////////////////////////////////////////////////////
// Private OpenCL solver class
///////////////////////////////////////////////////////////////////////////////

class GpuSolverOpenCl::Impl
{
public:

    Impl(const Mesh& mesh, const SolverParameters& parameters);

    ~Impl();

    void reset();

    void step(std::size_t num_iterations);

    void draw(sf::RenderTarget& target);

    void draw_velocity_field(sf::RenderTarget& target);

    float average_temperature() const;

    float max_displacement() const;

private:

    void add_heat(const std::size_t (&global)[2], const std::size_t (&local)[2]);

    void add_buoyancy(const std::size_t (&global)[2], const std::size_t (&local)[2]);

    void apply_velocity_boundaries(const std::size_t (&global)[2], const std::size_t (&local)[2]);

    void advect_velocity(const std::size_t (&global)[2], const std::size_t (&local)[2]);

    void diffuse_velocity(const std::size_t (&global)[2], const std::size_t (&local)[2]);

    void project_velocity(const std::size_t (&global)[2], const std::size_t (&local)[2]);

    void advect_temperature(const std::size_t (&global)[2], const std::size_t (&local)[2]);

    void diffuse_temperature(const std::size_t (&global)[2], const std::size_t (&local)[2]);

    void apply_temperature_boundaries(const std::size_t (&global)[2], const std::size_t (&local)[2]);

private:

    const Mesh& mesh_;

    const SolverParameters& parameters_;

    std::vector<std::uint8_t> pixels_;

    sf::Texture texture_;

    cl_device_id device_ = nullptr;

    cl_context context_ = nullptr;

    cl_command_queue command_queue_ = nullptr;

    cl_program program_ = nullptr;

    cl_kernel add_heat_kernel_ = nullptr;
    cl_kernel add_buoyancy_kernel_ = nullptr;
    cl_kernel apply_temperature_boundaries_kernel_ = nullptr;
    cl_kernel apply_velocity_boundaries_kernel_ = nullptr;
    cl_kernel advect_temperature_kernel_ = nullptr;
    cl_kernel advect_velocity_kernel_ = nullptr;
    cl_kernel diffuse_temperature_kernel_ = nullptr;
    cl_kernel diffuse_velocity_kernel_ = nullptr;
    cl_kernel compute_divergence_kernel_ = nullptr;
    cl_kernel clear_pressure_kernel_ = nullptr;
    cl_kernel solve_pressure_kernel_ = nullptr;
    cl_kernel subtract_pressure_gradient_kernel_ = nullptr;
    cl_kernel render_kernel_ = nullptr;

    cl_mem type_ = nullptr;
    cl_mem curr_temperature_ = nullptr;
    cl_mem next_temperature_ = nullptr;
    cl_mem curr_velocity_ = nullptr;
    cl_mem next_velocity_ = nullptr;
    cl_mem curr_pressure_ = nullptr;
    cl_mem next_pressure_ = nullptr;
    cl_mem divergence_ = nullptr;
    cl_mem pixels_buffer_ = nullptr;
};

///////////////////////////////////////////////////////////////////////////////
// Public member functions
///////////////////////////////////////////////////////////////////////////////

GpuSolverOpenCl::Impl::Impl(const Mesh& mesh, const SolverParameters& parameters)
    : mesh_(mesh)
    , parameters_(parameters)
    , pixels_(mesh.width() * mesh.height() * 4)
    , texture_(sf::Vector2u(mesh.width(), mesh.height()))
{
    device_ = choose_device();

    std::cout << "DEVICE: " << get_device_string(device_, CL_DEVICE_NAME) << std::endl;

    context_ = create_context(device_);

    command_queue_ = create_command_queue(device_, context_);

    program_ = create_and_build_program(device_, context_, reinterpret_cast<const char*>(src_opencl_solver_cl), src_opencl_solver_cl_len);

    add_heat_kernel_ = create_kernel(program_, "add_heat");
    add_buoyancy_kernel_ = create_kernel(program_, "add_buoyancy");
    apply_temperature_boundaries_kernel_ = create_kernel(program_, "apply_temperature_boundaries");
    apply_velocity_boundaries_kernel_ = create_kernel(program_, "apply_velocity_boundaries");
    advect_temperature_kernel_ = create_kernel(program_, "advect_temperature");
    advect_velocity_kernel_ = create_kernel(program_, "advect_velocity");
    diffuse_temperature_kernel_ = create_kernel(program_, "diffuse_temperature");
    diffuse_velocity_kernel_ = create_kernel(program_, "diffuse_velocity");
    compute_divergence_kernel_ = create_kernel(program_, "compute_divergence");
    clear_pressure_kernel_ = create_kernel(program_, "clear_pressure");
    solve_pressure_kernel_ = create_kernel(program_, "solve_pressure");
    subtract_pressure_gradient_kernel_ = create_kernel(program_, "subtract_pressure_gradient");
    render_kernel_ = create_kernel(program_, "render_heat");

    const std::size_t cell_count = mesh_.width() * mesh_.height();

    type_ = create_buffer(context_, CL_MEM_READ_ONLY, cell_count * sizeof(std::uint8_t));
    curr_temperature_ = create_buffer(context_, CL_MEM_READ_WRITE, cell_count * sizeof(float));
    next_temperature_ = create_buffer(context_, CL_MEM_READ_WRITE, cell_count * sizeof(float));
    curr_velocity_ = create_buffer(context_, CL_MEM_READ_WRITE, cell_count * sizeof(float) * 2);
    next_velocity_ = create_buffer(context_, CL_MEM_READ_WRITE, cell_count * sizeof(float) * 2);
    curr_pressure_ = create_buffer(context_, CL_MEM_READ_WRITE, cell_count * sizeof(float));
    next_pressure_ = create_buffer(context_, CL_MEM_READ_WRITE, cell_count * sizeof(float));
    divergence_ = create_buffer(context_, CL_MEM_READ_WRITE, cell_count * sizeof(float));
    pixels_buffer_ = create_buffer(context_, CL_MEM_WRITE_ONLY, cell_count * sizeof(std::uint8_t) * 4);

    reset();
}

GpuSolverOpenCl::Impl::~Impl()
{
    release_mem_object(pixels_buffer_);
    release_mem_object(divergence_);
    release_mem_object(next_pressure_);
    release_mem_object(curr_pressure_);
    release_mem_object(next_velocity_);
    release_mem_object(curr_velocity_);
    release_mem_object(next_temperature_);
    release_mem_object(curr_temperature_);
    release_mem_object(type_);
    release_kernel(render_kernel_);
    release_kernel(subtract_pressure_gradient_kernel_);
    release_kernel(solve_pressure_kernel_);
    release_kernel(clear_pressure_kernel_);
    release_kernel(compute_divergence_kernel_);
    release_kernel(diffuse_velocity_kernel_);
    release_kernel(diffuse_temperature_kernel_);
    release_kernel(advect_velocity_kernel_);
    release_kernel(advect_temperature_kernel_);
    release_kernel(apply_velocity_boundaries_kernel_);
    release_kernel(apply_temperature_boundaries_kernel_);
    release_kernel(add_buoyancy_kernel_);
    release_kernel(add_heat_kernel_);
    release_program(program_);
    release_command_queue(command_queue_);
    release_context(context_);
}

void GpuSolverOpenCl::Impl::reset()
{
    const std::size_t cell_count = mesh_.width() * mesh_.height();

    std::vector<std::uint8_t> cell_type(cell_count);
    std::vector<float> initial_temperature(cell_count);

    for (std::size_t i = 0; i < cell_count; ++i)
    {
        cell_type[i] = static_cast<std::uint8_t>(mesh_.type(i));
        initial_temperature[i] = mesh_.initial_temperature(i);
    }

    const std::array<float, 2> zero_vector = {0.0f, 0.0f};
    const float zero_scalar = 0.0f;

    check_cl
    (
        clEnqueueWriteBuffer
        (
            command_queue_,
            type_,
            CL_TRUE,
            0,
            cell_count * sizeof(std::uint8_t),
            cell_type.data(),
            0,
            nullptr,
            nullptr
        ),
        "clEnqueueWriteBuffer"
    );

    check_cl
    (
        clEnqueueWriteBuffer
        (
            command_queue_,
            curr_temperature_,
            CL_TRUE,
            0,
            cell_count * sizeof(float),
            initial_temperature.data(),
            0,
            nullptr,
            nullptr
        ),
        "clEnqueueWriteBuffer"
    );

    check_cl
    (
        clEnqueueWriteBuffer
        (
            command_queue_,
            next_temperature_,
            CL_TRUE,
            0,
            cell_count * sizeof(float),
            initial_temperature.data(),
            0,
            nullptr,
            nullptr
        ),
        "clEnqueueWriteBuffer"
    );

    check_cl
    (
        clEnqueueFillBuffer
        (
            command_queue_,
            curr_velocity_,
            &zero_vector,
            sizeof(float) * 2,
            0,
            cell_count * sizeof(float) * 2,
            0,
            nullptr,
            nullptr
        ),
        "clEnqueueFillBuffer"
    );

    check_cl
    (
        clEnqueueFillBuffer
        (
            command_queue_,
            next_velocity_,
            &zero_vector,
            sizeof(float) * 2,
            0,
            cell_count * sizeof(float) * 2,
            0,
            nullptr,
            nullptr
        ),
        "clEnqueueFillBuffer"
    );

    check_cl
    (
        clEnqueueFillBuffer
        (
            command_queue_,
            curr_pressure_,
            &zero_scalar,
            sizeof(float),
            0,
            cell_count * sizeof(float),
            0,
            nullptr,
            nullptr
        ),
        "clEnqueueFillBuffer"
    );

    check_cl
    (
        clEnqueueFillBuffer
        (
            command_queue_,
            next_pressure_,
            &zero_scalar,
            sizeof(float),
            0,
            cell_count * sizeof(float),
            0,
            nullptr,
            nullptr
        ),
        "clEnqueueFillBuffer"
    );

    check_cl
    (
        clEnqueueFillBuffer
        (
            command_queue_,
            divergence_,
            &zero_scalar,
            sizeof(float),
            0,
            cell_count * sizeof(float),
            0,
            nullptr,
            nullptr
        ),
        "clEnqueueFillBuffer"
    );

    check_cl(clFinish(command_queue_), "clFinish");
}

void GpuSolverOpenCl::Impl::step(std::size_t num_iterations)
{
    const std::size_t local[2] = {16, 16};
    const std::size_t global[2] =
    {
        round_up(mesh_.width(), local[0]),
        round_up(mesh_.height(), local[1])
    };

    while (num_iterations-- > 0)
    {
        add_heat(global, local);

        add_buoyancy(global, local);

        apply_velocity_boundaries(global, local);

        advect_velocity(global, local);

        apply_velocity_boundaries(global, local);

        diffuse_velocity(global, local);

        apply_velocity_boundaries(global, local);

        project_velocity(global, local);

        apply_velocity_boundaries(global, local);

        advect_temperature(global, local);

        diffuse_temperature(global, local);

        apply_temperature_boundaries(global, local);
    }

    check_cl(clFinish(command_queue_), "clFinish");
}

void GpuSolverOpenCl::Impl::draw(sf::RenderTarget& target)
{
    const std::size_t cell_count = mesh_.width() * mesh_.height();

    set_kernel_arg(render_kernel_, 0, type_);
    set_kernel_arg(render_kernel_, 1, curr_temperature_);
    set_kernel_arg(render_kernel_, 2, pixels_buffer_);
    set_kernel_arg(render_kernel_, 3, cl_uint(mesh_.width()));
    set_kernel_arg(render_kernel_, 4, cl_uint(mesh_.height()));
    set_kernel_arg(render_kernel_, 5, mesh_.min_temperature());
    set_kernel_arg(render_kernel_, 6, mesh_.max_temperature());

    const std::size_t local[2] = {16, 16};
    const std::size_t global[2] =
    {
        round_up(mesh_.width(), local[0]),
        round_up(mesh_.height(), local[1])
    };

    run_kernel(command_queue_, render_kernel_, global, local);

    check_cl
    (
        clEnqueueReadBuffer
        (
            command_queue_,
            pixels_buffer_,
            CL_TRUE,
            0,
            cell_count * sizeof(std::uint8_t) * 4,
            pixels_.data(),
            0,
            nullptr,
            nullptr
        ),
        "clEnqueueReadBuffer"
    );

    texture_.update(pixels_.data());

    target.draw(sf::Sprite(texture_));
}

void GpuSolverOpenCl::Impl::draw_velocity_field(sf::RenderTarget& target)
{
    const std::size_t cell_count = mesh_.width() * mesh_.height();
    std::vector<cl_float2> velocity(cell_count);

    check_cl
    (
        clEnqueueReadBuffer
        (
            command_queue_,
            curr_velocity_,
            CL_TRUE,
            0,
            cell_count * sizeof(float) * 2,
            velocity.data(),
            0,
            nullptr,
            nullptr
        ),
        "clEnqueueReadBuffer"
    );

    sf::VertexArray lines(sf::PrimitiveType::Lines);

    for (std::size_t y = 0; y < mesh_.height(); y += 2)
    {
        for (std::size_t x = 0; x < mesh_.width(); x += 2)
        {
            const auto i = mesh_.index(x, y);
            const auto ti = mesh_.type(i);

            if (ti == CellType::Fluid)
            {
                const float vx = velocity[i].s[0];
                const float vy = velocity[i].s[1];
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

float GpuSolverOpenCl::Impl::average_temperature() const
{
    const std::size_t cell_count = mesh_.width() * mesh_.height();
    std::vector<float> temperature(cell_count);

    check_cl
    (
        clEnqueueReadBuffer
        (
            command_queue_,
            curr_temperature_,
            CL_TRUE,
            0,
            cell_count * sizeof(float),
            temperature.data(),
            0,
            nullptr,
            nullptr
        ),
        "clEnqueueReadBuffer"
    );

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

    return sum / count;
}

float GpuSolverOpenCl::Impl::max_displacement() const
{
    const std::size_t cell_count = mesh_.width() * mesh_.height();
    std::vector<cl_float2> velocity(cell_count);

    check_cl
    (
        clEnqueueReadBuffer
        (
            command_queue_,
            curr_velocity_,
            CL_TRUE,
            0,
            cell_count * sizeof(float) * 2,
            velocity.data(),
            0,
            nullptr,
            nullptr
        ),
        "clEnqueueReadBuffer"
    );

    float max_speed = 0.0f;

    for (std::size_t i = 0; i < cell_count; ++i)
    {
        const auto ti = mesh_.type(i);

        if (ti == CellType::Fluid)
        {
            const float vx = velocity[i].s[0];
            const float vy = velocity[i].s[1];
            const float speed = std::sqrt(vx * vx + vy * vy);

            if (speed > max_speed)
            {
                max_speed = speed;
            }
        }
    }

    return max_speed * parameters_.dt;
}

///////////////////////////////////////////////////////////////////////////////
// Private member functions
///////////////////////////////////////////////////////////////////////////////

void GpuSolverOpenCl::Impl::add_heat(const std::size_t (&global)[2], const std::size_t (&local)[2])
{
    set_kernel_arg(add_heat_kernel_, 0, type_);
    set_kernel_arg(add_heat_kernel_, 1, curr_temperature_);
    set_kernel_arg(add_heat_kernel_, 2, cl_uint(mesh_.width()));
    set_kernel_arg(add_heat_kernel_, 3, cl_uint(mesh_.height()));
    set_kernel_arg(add_heat_kernel_, 4, parameters_.source_heat_transfer);
    set_kernel_arg(add_heat_kernel_, 5, parameters_.dt);
    set_kernel_arg(add_heat_kernel_, 6, mesh_.max_temperature());

    run_kernel(command_queue_, add_heat_kernel_, global, local);
}

void GpuSolverOpenCl::Impl::add_buoyancy(const std::size_t (&global)[2], const std::size_t (&local)[2])
{
    set_kernel_arg(add_buoyancy_kernel_, 0, type_);
    set_kernel_arg(add_buoyancy_kernel_, 1, curr_temperature_);
    set_kernel_arg(add_buoyancy_kernel_, 2, curr_velocity_);
    set_kernel_arg(add_buoyancy_kernel_, 3, cl_uint(mesh_.width()));
    set_kernel_arg(add_buoyancy_kernel_, 4, cl_uint(mesh_.height()));
    set_kernel_arg(add_buoyancy_kernel_, 5, mesh_.min_temperature());
    set_kernel_arg(add_buoyancy_kernel_, 6, parameters_.buoyancy);
    set_kernel_arg(add_buoyancy_kernel_, 7, parameters_.dt);

    run_kernel(command_queue_, add_buoyancy_kernel_, global, local);
}

void GpuSolverOpenCl::Impl::apply_velocity_boundaries(const std::size_t (&global)[2], const std::size_t (&local)[2])
{
    set_kernel_arg(apply_velocity_boundaries_kernel_, 0, type_);
    set_kernel_arg(apply_velocity_boundaries_kernel_, 1, curr_velocity_);
    set_kernel_arg(apply_velocity_boundaries_kernel_, 2, cl_uint(mesh_.width()));
    set_kernel_arg(apply_velocity_boundaries_kernel_, 3, cl_uint(mesh_.height()));

    run_kernel(command_queue_, apply_velocity_boundaries_kernel_, global, local);
}

void GpuSolverOpenCl::Impl::advect_velocity(const std::size_t (&global)[2], const std::size_t (&local)[2])
{
    set_kernel_arg(advect_velocity_kernel_, 0, type_);
    set_kernel_arg(advect_velocity_kernel_, 1, curr_velocity_);
    set_kernel_arg(advect_velocity_kernel_, 2, next_velocity_);
    set_kernel_arg(advect_velocity_kernel_, 3, cl_uint(mesh_.width()));
    set_kernel_arg(advect_velocity_kernel_, 4, cl_uint(mesh_.height()));
    set_kernel_arg(advect_velocity_kernel_, 5, parameters_.dt);
    set_kernel_arg(advect_velocity_kernel_, 6, parameters_.velocity_damping);

    run_kernel(command_queue_, advect_velocity_kernel_, global, local);
    std::swap(curr_velocity_, next_velocity_);
}

void GpuSolverOpenCl::Impl::diffuse_velocity(const std::size_t (&global)[2], const std::size_t (&local)[2])
{
    set_kernel_arg(diffuse_velocity_kernel_, 0, type_);
    set_kernel_arg(diffuse_velocity_kernel_, 1, curr_velocity_);
    set_kernel_arg(diffuse_velocity_kernel_, 2, next_velocity_);
    set_kernel_arg(diffuse_velocity_kernel_, 3, cl_uint(mesh_.width()));
    set_kernel_arg(diffuse_velocity_kernel_, 4, cl_uint(mesh_.height()));
    set_kernel_arg(diffuse_velocity_kernel_, 5, parameters_.viscosity * parameters_.dt);

    run_kernel(command_queue_, diffuse_velocity_kernel_, global, local);
    std::swap(curr_velocity_, next_velocity_);
}

void GpuSolverOpenCl::Impl::project_velocity(const std::size_t (&global)[2], const std::size_t (&local)[2])
{
    set_kernel_arg(compute_divergence_kernel_, 0, type_);
    set_kernel_arg(compute_divergence_kernel_, 1, curr_velocity_);
    set_kernel_arg(compute_divergence_kernel_, 2, divergence_);
    set_kernel_arg(compute_divergence_kernel_, 3, cl_uint(mesh_.width()));
    set_kernel_arg(compute_divergence_kernel_, 4, cl_uint(mesh_.height()));
    run_kernel(command_queue_, compute_divergence_kernel_, global, local);

    set_kernel_arg(clear_pressure_kernel_, 0, curr_pressure_);
    set_kernel_arg(clear_pressure_kernel_, 1, next_pressure_);
    set_kernel_arg(clear_pressure_kernel_, 2, cl_uint(mesh_.width()));
    set_kernel_arg(clear_pressure_kernel_, 3, cl_uint(mesh_.height()));
    run_kernel(command_queue_, clear_pressure_kernel_, global, local);

    set_kernel_arg(solve_pressure_kernel_, 0, type_);
    set_kernel_arg(solve_pressure_kernel_, 3, divergence_);
    set_kernel_arg(solve_pressure_kernel_, 4, cl_uint(mesh_.width()));
    set_kernel_arg(solve_pressure_kernel_, 5, cl_uint(mesh_.height()));

    for (int iter = 0; iter < parameters_.pressure_iterations; ++iter)
    {
        set_kernel_arg(solve_pressure_kernel_, 1, curr_pressure_);
        set_kernel_arg(solve_pressure_kernel_, 2, next_pressure_);
        run_kernel(command_queue_, solve_pressure_kernel_, global, local);
        std::swap(curr_pressure_, next_pressure_);
    }

    set_kernel_arg(subtract_pressure_gradient_kernel_, 0, type_);
    set_kernel_arg(subtract_pressure_gradient_kernel_, 1, curr_pressure_);
    set_kernel_arg(subtract_pressure_gradient_kernel_, 2, curr_velocity_);
    set_kernel_arg(subtract_pressure_gradient_kernel_, 3, cl_uint(mesh_.width()));
    set_kernel_arg(subtract_pressure_gradient_kernel_, 4, cl_uint(mesh_.height()));

    run_kernel(command_queue_, subtract_pressure_gradient_kernel_, global, local);
}

void GpuSolverOpenCl::Impl::advect_temperature(const std::size_t (&global)[2], const std::size_t (&local)[2])
{
    set_kernel_arg(advect_temperature_kernel_, 0, type_);
    set_kernel_arg(advect_temperature_kernel_, 1, curr_temperature_);
    set_kernel_arg(advect_temperature_kernel_, 2, next_temperature_);
    set_kernel_arg(advect_temperature_kernel_, 3, curr_velocity_);
    set_kernel_arg(advect_temperature_kernel_, 4, cl_uint(mesh_.width()));
    set_kernel_arg(advect_temperature_kernel_, 5, cl_uint(mesh_.height()));
    set_kernel_arg(advect_temperature_kernel_, 6, parameters_.dt);

    run_kernel(command_queue_, advect_temperature_kernel_, global, local);
    std::swap(curr_temperature_, next_temperature_);
}

void GpuSolverOpenCl::Impl::diffuse_temperature(const std::size_t (&global)[2], const std::size_t (&local)[2])
{
    set_kernel_arg(diffuse_temperature_kernel_, 0, type_);
    set_kernel_arg(diffuse_temperature_kernel_, 1, curr_temperature_);
    set_kernel_arg(diffuse_temperature_kernel_, 2, next_temperature_);
    set_kernel_arg(diffuse_temperature_kernel_, 3, cl_uint(mesh_.width()));
    set_kernel_arg(diffuse_temperature_kernel_, 4, cl_uint(mesh_.height()));
    set_kernel_arg(diffuse_temperature_kernel_, 5, parameters_.thermal_diffusion * parameters_.dt);

    run_kernel(command_queue_, diffuse_temperature_kernel_, global, local);
    std::swap(curr_temperature_, next_temperature_);
}

void GpuSolverOpenCl::Impl::apply_temperature_boundaries(const std::size_t (&global)[2], const std::size_t (&local)[2])
{
    set_kernel_arg(apply_temperature_boundaries_kernel_, 0, type_);
    set_kernel_arg(apply_temperature_boundaries_kernel_, 1, curr_temperature_);
    set_kernel_arg(apply_temperature_boundaries_kernel_, 2, cl_uint(mesh_.width()));
    set_kernel_arg(apply_temperature_boundaries_kernel_, 3, cl_uint(mesh_.height()));
    set_kernel_arg(apply_temperature_boundaries_kernel_, 4, mesh_.min_temperature());
    set_kernel_arg(apply_temperature_boundaries_kernel_, 5, mesh_.max_temperature());

    run_kernel(command_queue_, apply_temperature_boundaries_kernel_, global, local);
}

///////////////////////////////////////////////////////////////////////////////
// Public OpenCL solver class
///////////////////////////////////////////////////////////////////////////////

GpuSolverOpenCl::GpuSolverOpenCl(const Mesh& mesh, const SolverParameters& parameters)
    : impl_(std::make_unique<Impl>(mesh, parameters))
{}

GpuSolverOpenCl::~GpuSolverOpenCl() = default;

void GpuSolverOpenCl::reset()
{
    impl_->reset();
}

void GpuSolverOpenCl::step(std::size_t num_iterations)
{
    impl_->step(num_iterations);
}

void GpuSolverOpenCl::draw(sf::RenderTarget& target)
{
    impl_->draw(target);
}

void GpuSolverOpenCl::draw_velocity_field(sf::RenderTarget& target)
{
    impl_->draw_velocity_field(target);
}

float GpuSolverOpenCl::average_temperature() const
{
    return impl_->average_temperature();
}

float GpuSolverOpenCl::max_displacement() const
{
    return impl_->max_displacement();
}
