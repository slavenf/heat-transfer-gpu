BUILD_DIR := ./build

TARGET := heat-transfer

SRCS := $(wildcard src/*.cpp) \
        ./third-party/glad/src/glad.c
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:%.o=%.d)

GLSL_SRCS := $(wildcard src/*.glsl)
GLSL_OBJS := $(GLSL_SRCS:%=$(BUILD_DIR)/%.o)

OPENCL_SRCS := $(wildcard src/*.cl)
OPENCL_OBJS := $(OPENCL_SRCS:%=$(BUILD_DIR)/%.o)

SYSTEM_INCLUDES := ./third-party/cxxopts/include \
                   ./third-party/glad/include \
                   ./third-party/nlohmann-json/include

CPPFLAGS := -MMD -MP ${addprefix -isystem, $(SYSTEM_INCLUDES)}
CXXFLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wfatal-errors -fopenmp
LDFLAGS  := -fopenmp
LDLIBS   := -lsfml-graphics -lsfml-window -lsfml-system -lOpenCL

# Build target
$(BUILD_DIR)/$(TARGET): $(OBJS) $(GLSL_OBJS) $(OPENCL_OBJS)
	mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Compile C sources
$(BUILD_DIR)/%.c.o: %.c
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ -c $<

# Compile C++ sources
$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ -c $<

# Compile GLSL shaders
$(BUILD_DIR)/%.glsl.o: %.glsl
	mkdir -p $(dir $@)
	xxd -i $< $(BUILD_DIR)/$*.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ -c $(BUILD_DIR)/$*.cpp

# Compile OpenCL sources
$(BUILD_DIR)/%.cl.o: %.cl
	mkdir -p $(dir $@)
	xxd -i $< $(BUILD_DIR)/$*.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ -c $(BUILD_DIR)/$*.cpp

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

-include $(DEPS)
