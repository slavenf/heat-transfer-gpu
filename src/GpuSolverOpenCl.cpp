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

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include "GpuSolverOpenCl.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

static cl_mem create_buffer
(
    cl_context context,
    cl_mem_flags flags,
    std::size_t size,
    void* host_ptr
)
{
    cl_int error = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(context, flags, size, host_ptr, &error);
    check_cl(error, "clCreateBuffer");
    return buffer;
}

static cl_program build_program(cl_context context, cl_device_id device, const char* source, std::size_t source_length)
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

template <typename T>
static void set_kernel_arg(cl_kernel kernel, cl_uint index, const T& value)
{
    check_cl(clSetKernelArg(kernel, index, sizeof(T), &value), "clSetKernelArg");
}

static std::size_t round_up(std::size_t value, std::size_t multiple)
{
    return ((value + multiple - 1) / multiple) * multiple;
}

static std::vector<float> create_initial_temperature(const Mesh& mesh)
{
    std::vector<float> initial_temperature(mesh.width() * mesh.height());

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
    std::vector<std::uint8_t> cell_type(mesh.width() * mesh.height());

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

struct GpuSolverOpenCl::Impl
{
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel step_kernel = nullptr;
    cl_kernel render_kernel = nullptr;
    cl_mem type = nullptr;
    cl_mem curr = nullptr;
    cl_mem next = nullptr;
    cl_mem pixels = nullptr;

    ~Impl()
    {
        if (pixels != nullptr)
        {
            clReleaseMemObject(pixels);
        }

        if (next != nullptr)
        {
            clReleaseMemObject(next);
        }

        if (curr != nullptr)
        {
            clReleaseMemObject(curr);
        }

        if (type != nullptr)
        {
            clReleaseMemObject(type);
        }

        if (render_kernel != nullptr)
        {
            clReleaseKernel(render_kernel);
        }

        if (step_kernel != nullptr)
        {
            clReleaseKernel(step_kernel);
        }

        if (program != nullptr)
        {
            clReleaseProgram(program);
        }

        if (queue != nullptr)
        {
            clReleaseCommandQueue(queue);
        }

        if (context != nullptr)
        {
            clReleaseContext(context);
        }
    }
};

GpuSolverOpenCl::GpuSolverOpenCl(const Mesh& mesh)
    : mesh_(mesh)
    , pixels_(mesh.width() * mesh.height() * 4)
    , texture_(sf::Vector2u(mesh.width(), mesh.height()))
    , impl_(std::make_unique<Impl>())
{
    std::vector<std::uint8_t> cell_type = create_cell_type(mesh_);
    std::vector<float> initial_temperature = create_initial_temperature(mesh_);

    impl_->device = choose_device();

    cl_int error = CL_SUCCESS;
    impl_->context = clCreateContext(nullptr, 1, &impl_->device, nullptr, nullptr, &error);
    check_cl(error, "clCreateContext");

    impl_->queue = clCreateCommandQueue(impl_->context, impl_->device, 0, &error);
    check_cl(error, "clCreateCommandQueue");

    extern unsigned char src_opencl_solver_cl[];
    extern unsigned int src_opencl_solver_cl_len;
    impl_->program = build_program(impl_->context, impl_->device, reinterpret_cast<const char*>(src_opencl_solver_cl), src_opencl_solver_cl_len);

    impl_->step_kernel = clCreateKernel(impl_->program, "step_heat", &error);
    check_cl(error, "clCreateKernel");

    impl_->render_kernel = clCreateKernel(impl_->program, "render_heat", &error);
    check_cl(error, "clCreateKernel");

    impl_->type = create_buffer
    (
        impl_->context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        cell_type.size() * sizeof(cell_type[0]),
        cell_type.data()
    );

    impl_->curr = create_buffer
    (
        impl_->context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        initial_temperature.size() * sizeof(initial_temperature[0]),
        initial_temperature.data()
    );

    impl_->next = create_buffer
    (
        impl_->context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        initial_temperature.size() * sizeof(initial_temperature[0]),
        initial_temperature.data()
    );

    impl_->pixels = create_buffer
    (
        impl_->context,
        CL_MEM_WRITE_ONLY,
        pixels_.size() * sizeof(pixels_[0]),
        nullptr
    );

    std::cout << "DEVICE: " << get_device_string(impl_->device, CL_DEVICE_NAME) << std::endl;
}

GpuSolverOpenCl::~GpuSolverOpenCl() = default;

void GpuSolverOpenCl::reset()
{
    const std::vector<float> initial_temperature = create_initial_temperature(mesh_);

    check_cl
    (
        clEnqueueWriteBuffer
        (
            impl_->queue,
            impl_->curr,
            CL_TRUE,
            0,
            initial_temperature.size() * sizeof(initial_temperature[0]),
            initial_temperature.data(),
            0,
            nullptr,
            nullptr
        ),
        "clEnqueueWriteBuffer"
    );
}

void GpuSolverOpenCl::step(std::size_t num_iterations)
{
    const cl_uint width = static_cast<cl_uint>(mesh_.width());
    const cl_uint height = static_cast<cl_uint>(mesh_.height());

    set_kernel_arg(impl_->step_kernel, 0, impl_->type);
    set_kernel_arg(impl_->step_kernel, 1, impl_->curr);
    set_kernel_arg(impl_->step_kernel, 2, impl_->next);
    set_kernel_arg(impl_->step_kernel, 3, width);
    set_kernel_arg(impl_->step_kernel, 4, height);
    set_kernel_arg(impl_->step_kernel, 5, diffusion_);

    const std::size_t local[2] = {16, 16};
    const std::size_t global[2] =
    {
        round_up(mesh_.width(), local[0]),
        round_up(mesh_.height(), local[1])
    };

    while (num_iterations-- > 0)
    {
        check_cl
        (
            clEnqueueNDRangeKernel
            (
                impl_->queue,
                impl_->step_kernel,
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

        std::swap(impl_->curr, impl_->next);
        set_kernel_arg(impl_->step_kernel, 1, impl_->curr);
        set_kernel_arg(impl_->step_kernel, 2, impl_->next);
    }

    check_cl(clFinish(impl_->queue), "clFinish");
}

void GpuSolverOpenCl::draw(sf::RenderTarget& target)
{
    const cl_uint width = static_cast<cl_uint>(mesh_.width());
    const cl_uint height = static_cast<cl_uint>(mesh_.height());

    constexpr float min_temp = 0.0f;
    constexpr float max_temp = 100.0f;

    set_kernel_arg(impl_->render_kernel, 0, impl_->type);
    set_kernel_arg(impl_->render_kernel, 1, impl_->curr);
    set_kernel_arg(impl_->render_kernel, 2, impl_->pixels);
    set_kernel_arg(impl_->render_kernel, 3, width);
    set_kernel_arg(impl_->render_kernel, 4, height);
    set_kernel_arg(impl_->render_kernel, 5, min_temp);
    set_kernel_arg(impl_->render_kernel, 6, max_temp);

    const std::size_t local[2] = {16, 16};
    const std::size_t global[2] =
    {
        round_up(mesh_.width(), local[0]),
        round_up(mesh_.height(), local[1])
    };

    check_cl
    (
        clEnqueueNDRangeKernel(impl_->queue, impl_->render_kernel, 2, nullptr, global, local, 0, nullptr, nullptr),
        "clEnqueueNDRangeKernel"
    );

    check_cl
    (
        clEnqueueReadBuffer
        (
            impl_->queue,
            impl_->pixels,
            CL_TRUE,
            0,
            pixels_.size() * sizeof(pixels_[0]),
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
