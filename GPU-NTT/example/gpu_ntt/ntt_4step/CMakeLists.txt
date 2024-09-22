set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION})

if(NOT DEFINED MODULAR_REDUCTION_TYPE)
    set(MODULAR_REDUCTION_TYPE 0)
endif()

add_executable(gpu_4step_ntt_examples test_4step_ntt.cu) #定义可执行文件名，指定生成的可执行文件名和对应的源文件 add_executable 用于创建可执行文件，它是最终用户可以运行的应用程序。
add_executable(gpu_4step_intt_examples test_4step_intt.cu)
add_executable(gpu_4step_negative_ntt_examples test_4step_negative_ntt.cu)
add_executable(gpu_4step_negative_intt_examples test_4step_negative_intt.cu)


if(MODULAR_REDUCTION_TYPE EQUAL 0)
    add_compile_definitions(BARRETT_64)
    target_compile_definitions(gpu_4step_ntt_examples PRIVATE BARRETT_64)
    target_compile_definitions(gpu_4step_intt_examples PRIVATE BARRETT_64)
    target_compile_definitions(gpu_4step_negative_ntt_examples PRIVATE BARRETT_64)
    target_compile_definitions(gpu_4step_negative_intt_examples PRIVATE BARRETT_64)
elseif(MODULAR_REDUCTION_TYPE EQUAL 1)
    add_compile_definitions(GOLDILOCKS_64)
    target_compile_definitions(gpu_4step_ntt_examples PRIVATE GOLDILOCKS_64)
    target_compile_definitions(gpu_4step_intt_examples PRIVATE GOLDILOCKS_64)
    target_compile_definitions(gpu_4step_negative_ntt_examples  PRIVATE GOLDILOCKS_64)
    target_compile_definitions(gpu_4step_negative_intt_examples  PRIVATE GOLDILOCKS_64)
elseif(MODULAR_REDUCTION_TYPE EQUAL 2)
    add_compile_definitions(PLANTARD_64)
    target_compile_definitions(gpu_4step_ntt_examples PRIVATE PLANTARD_64)
    target_compile_definitions(gpu_4step_intt_examples PRIVATE PLANTARD_64)
    target_compile_definitions(gpu_4step_negative_ntt_examples PRIVATE PLANTARD_64)
    target_compile_definitions(gpu_4step_negative_intt_examples PRIVATE PLANTARD_64)
else()
    message(FATAL_ERROR "Please use valid MODULAR_REDUCTION_TYPE! (0,1,2)")
endif()

target_link_libraries(gpu_4step_ntt_examples PRIVATE ntt ntt_4step) #链接上生成的ntt和ntt_4step库

set_target_properties(gpu_4step_ntt_examples PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(gpu_4step_ntt_examples PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include) #TODO:这里的include是否include到了正确的头文件夹下呢？


target_link_libraries(gpu_4step_intt_examples PRIVATE ntt ntt_4step)

set_target_properties(gpu_4step_intt_examples PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(gpu_4step_intt_examples PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)


target_link_libraries(gpu_4step_negative_ntt_examples PRIVATE ntt ntt_4step)

set_target_properties(gpu_4step_negative_ntt_examples PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(gpu_4step_negative_ntt_examples PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)


target_link_libraries(gpu_4step_negative_intt_examples PRIVATE ntt ntt_4step)

set_target_properties(gpu_4step_negative_intt_examples PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(gpu_4step_negative_intt_examples PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)