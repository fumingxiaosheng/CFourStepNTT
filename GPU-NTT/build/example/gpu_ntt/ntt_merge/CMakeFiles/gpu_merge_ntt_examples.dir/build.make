# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hxw/CUDA/4-step/GPU-NTT

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hxw/CUDA/4-step/GPU-NTT/build

# Include any dependencies generated for this target.
include example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/compiler_depend.make

# Include the progress variables for this target.
include example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/progress.make

# Include the compile flags for this target's objects.
include example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/flags.make

example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/test_merge_ntt.cu.o: example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/flags.make
example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/test_merge_ntt.cu.o: ../example/gpu_ntt/ntt_merge/test_merge_ntt.cu
example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/test_merge_ntt.cu.o: example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hxw/CUDA/4-step/GPU-NTT/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/test_merge_ntt.cu.o"
	cd /home/hxw/CUDA/4-step/GPU-NTT/build/example/gpu_ntt/ntt_merge && /usr/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/test_merge_ntt.cu.o -MF CMakeFiles/gpu_merge_ntt_examples.dir/test_merge_ntt.cu.o.d -x cu -dc /home/hxw/CUDA/4-step/GPU-NTT/example/gpu_ntt/ntt_merge/test_merge_ntt.cu -o CMakeFiles/gpu_merge_ntt_examples.dir/test_merge_ntt.cu.o

example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/test_merge_ntt.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/gpu_merge_ntt_examples.dir/test_merge_ntt.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/test_merge_ntt.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/gpu_merge_ntt_examples.dir/test_merge_ntt.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target gpu_merge_ntt_examples
gpu_merge_ntt_examples_OBJECTS = \
"CMakeFiles/gpu_merge_ntt_examples.dir/test_merge_ntt.cu.o"

# External object files for target gpu_merge_ntt_examples
gpu_merge_ntt_examples_EXTERNAL_OBJECTS =

example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/cmake_device_link.o: example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/test_merge_ntt.cu.o
example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/cmake_device_link.o: example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/build.make
example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/cmake_device_link.o: src/ntt/ntt_merge/lib/libntt.a
example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/cmake_device_link.o: src/common/lib/libcommon.a
example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/cmake_device_link.o: example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hxw/CUDA/4-step/GPU-NTT/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/gpu_merge_ntt_examples.dir/cmake_device_link.o"
	cd /home/hxw/CUDA/4-step/GPU-NTT/build/example/gpu_ntt/ntt_merge && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gpu_merge_ntt_examples.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/build: example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/cmake_device_link.o
.PHONY : example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/build

# Object files for target gpu_merge_ntt_examples
gpu_merge_ntt_examples_OBJECTS = \
"CMakeFiles/gpu_merge_ntt_examples.dir/test_merge_ntt.cu.o"

# External object files for target gpu_merge_ntt_examples
gpu_merge_ntt_examples_EXTERNAL_OBJECTS =

bin/gpu_merge_ntt_examples: example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/test_merge_ntt.cu.o
bin/gpu_merge_ntt_examples: example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/build.make
bin/gpu_merge_ntt_examples: src/ntt/ntt_merge/lib/libntt.a
bin/gpu_merge_ntt_examples: src/common/lib/libcommon.a
bin/gpu_merge_ntt_examples: example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/cmake_device_link.o
bin/gpu_merge_ntt_examples: example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hxw/CUDA/4-step/GPU-NTT/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable ../../../bin/gpu_merge_ntt_examples"
	cd /home/hxw/CUDA/4-step/GPU-NTT/build/example/gpu_ntt/ntt_merge && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gpu_merge_ntt_examples.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/build: bin/gpu_merge_ntt_examples
.PHONY : example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/build

example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/clean:
	cd /home/hxw/CUDA/4-step/GPU-NTT/build/example/gpu_ntt/ntt_merge && $(CMAKE_COMMAND) -P CMakeFiles/gpu_merge_ntt_examples.dir/cmake_clean.cmake
.PHONY : example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/clean

example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/depend:
	cd /home/hxw/CUDA/4-step/GPU-NTT/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hxw/CUDA/4-step/GPU-NTT /home/hxw/CUDA/4-step/GPU-NTT/example/gpu_ntt/ntt_merge /home/hxw/CUDA/4-step/GPU-NTT/build /home/hxw/CUDA/4-step/GPU-NTT/build/example/gpu_ntt/ntt_merge /home/hxw/CUDA/4-step/GPU-NTT/build/example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : example/gpu_ntt/ntt_merge/CMakeFiles/gpu_merge_ntt_examples.dir/depend

