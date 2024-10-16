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
include benchmark/CMakeFiles/benchmark_merge_ntt.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include benchmark/CMakeFiles/benchmark_merge_ntt.dir/compiler_depend.make

# Include the progress variables for this target.
include benchmark/CMakeFiles/benchmark_merge_ntt.dir/progress.make

# Include the compile flags for this target's objects.
include benchmark/CMakeFiles/benchmark_merge_ntt.dir/flags.make

benchmark/CMakeFiles/benchmark_merge_ntt.dir/bench_merge_ntt.cu.o: benchmark/CMakeFiles/benchmark_merge_ntt.dir/flags.make
benchmark/CMakeFiles/benchmark_merge_ntt.dir/bench_merge_ntt.cu.o: ../benchmark/bench_merge_ntt.cu
benchmark/CMakeFiles/benchmark_merge_ntt.dir/bench_merge_ntt.cu.o: benchmark/CMakeFiles/benchmark_merge_ntt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hxw/CUDA/4-step/GPU-NTT/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object benchmark/CMakeFiles/benchmark_merge_ntt.dir/bench_merge_ntt.cu.o"
	cd /home/hxw/CUDA/4-step/GPU-NTT/build/benchmark && /usr/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT benchmark/CMakeFiles/benchmark_merge_ntt.dir/bench_merge_ntt.cu.o -MF CMakeFiles/benchmark_merge_ntt.dir/bench_merge_ntt.cu.o.d -x cu -dc /home/hxw/CUDA/4-step/GPU-NTT/benchmark/bench_merge_ntt.cu -o CMakeFiles/benchmark_merge_ntt.dir/bench_merge_ntt.cu.o

benchmark/CMakeFiles/benchmark_merge_ntt.dir/bench_merge_ntt.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/benchmark_merge_ntt.dir/bench_merge_ntt.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

benchmark/CMakeFiles/benchmark_merge_ntt.dir/bench_merge_ntt.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/benchmark_merge_ntt.dir/bench_merge_ntt.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target benchmark_merge_ntt
benchmark_merge_ntt_OBJECTS = \
"CMakeFiles/benchmark_merge_ntt.dir/bench_merge_ntt.cu.o"

# External object files for target benchmark_merge_ntt
benchmark_merge_ntt_EXTERNAL_OBJECTS =

benchmark/CMakeFiles/benchmark_merge_ntt.dir/cmake_device_link.o: benchmark/CMakeFiles/benchmark_merge_ntt.dir/bench_merge_ntt.cu.o
benchmark/CMakeFiles/benchmark_merge_ntt.dir/cmake_device_link.o: benchmark/CMakeFiles/benchmark_merge_ntt.dir/build.make
benchmark/CMakeFiles/benchmark_merge_ntt.dir/cmake_device_link.o: src/ntt/ntt_merge/lib/libntt.a
benchmark/CMakeFiles/benchmark_merge_ntt.dir/cmake_device_link.o: src/common/lib/libcommon.a
benchmark/CMakeFiles/benchmark_merge_ntt.dir/cmake_device_link.o: benchmark/CMakeFiles/benchmark_merge_ntt.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hxw/CUDA/4-step/GPU-NTT/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/benchmark_merge_ntt.dir/cmake_device_link.o"
	cd /home/hxw/CUDA/4-step/GPU-NTT/build/benchmark && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/benchmark_merge_ntt.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
benchmark/CMakeFiles/benchmark_merge_ntt.dir/build: benchmark/CMakeFiles/benchmark_merge_ntt.dir/cmake_device_link.o
.PHONY : benchmark/CMakeFiles/benchmark_merge_ntt.dir/build

# Object files for target benchmark_merge_ntt
benchmark_merge_ntt_OBJECTS = \
"CMakeFiles/benchmark_merge_ntt.dir/bench_merge_ntt.cu.o"

# External object files for target benchmark_merge_ntt
benchmark_merge_ntt_EXTERNAL_OBJECTS =

bin/benchmark_merge_ntt: benchmark/CMakeFiles/benchmark_merge_ntt.dir/bench_merge_ntt.cu.o
bin/benchmark_merge_ntt: benchmark/CMakeFiles/benchmark_merge_ntt.dir/build.make
bin/benchmark_merge_ntt: src/ntt/ntt_merge/lib/libntt.a
bin/benchmark_merge_ntt: src/common/lib/libcommon.a
bin/benchmark_merge_ntt: benchmark/CMakeFiles/benchmark_merge_ntt.dir/cmake_device_link.o
bin/benchmark_merge_ntt: benchmark/CMakeFiles/benchmark_merge_ntt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hxw/CUDA/4-step/GPU-NTT/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable ../bin/benchmark_merge_ntt"
	cd /home/hxw/CUDA/4-step/GPU-NTT/build/benchmark && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/benchmark_merge_ntt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
benchmark/CMakeFiles/benchmark_merge_ntt.dir/build: bin/benchmark_merge_ntt
.PHONY : benchmark/CMakeFiles/benchmark_merge_ntt.dir/build

benchmark/CMakeFiles/benchmark_merge_ntt.dir/clean:
	cd /home/hxw/CUDA/4-step/GPU-NTT/build/benchmark && $(CMAKE_COMMAND) -P CMakeFiles/benchmark_merge_ntt.dir/cmake_clean.cmake
.PHONY : benchmark/CMakeFiles/benchmark_merge_ntt.dir/clean

benchmark/CMakeFiles/benchmark_merge_ntt.dir/depend:
	cd /home/hxw/CUDA/4-step/GPU-NTT/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hxw/CUDA/4-step/GPU-NTT /home/hxw/CUDA/4-step/GPU-NTT/benchmark /home/hxw/CUDA/4-step/GPU-NTT/build /home/hxw/CUDA/4-step/GPU-NTT/build/benchmark /home/hxw/CUDA/4-step/GPU-NTT/build/benchmark/CMakeFiles/benchmark_merge_ntt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : benchmark/CMakeFiles/benchmark_merge_ntt.dir/depend

