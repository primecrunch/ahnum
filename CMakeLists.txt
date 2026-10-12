cmake_minimum_required(VERSION 3.20)
project(ahnum C ASM)

set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS_RELEASE "-O3 -mcpu=apple-m1")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_path(GMP_INCLUDE gmp.h HINTS /opt/homebrew/opt/gmp/include)
find_library(GMP_LIB gmp HINTS /opt/homebrew/opt/gmp/lib)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(ASM_DIR ${CMAKE_BINARY_DIR}/asm)
set(GEN_ASM ${ASM_DIR}/fft_stages.S ${ASM_DIR}/norm.S ${ASM_DIR}/pass2.S ${ASM_DIR}/sme.S)
add_custom_command(
  OUTPUT ${GEN_ASM}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/gen/gen_kernels.py ${ASM_DIR}
  DEPENDS ${CMAKE_SOURCE_DIR}/gen/gen_kernels.py
  COMMENT "Generating NEON kernels")

add_library(ahnum STATIC
  ${GEN_ASM}
  src/ctx.c
  src/ref_fft.c
  src/fast2p.c
  src/neon.c
  src/prp.c
  src/threads.c
  src/md5.c
  src/prstproof.c
)
target_include_directories(ahnum PUBLIC include PRIVATE src)

add_executable(gmp_ref tools/gmp_ref.c)
target_include_directories(gmp_ref PRIVATE ${GMP_INCLUDE})
target_link_libraries(gmp_ref ${GMP_LIB})

add_executable(test_gmp tests/test_gmp.c)
target_include_directories(test_gmp PRIVATE ${GMP_INCLUDE} src)
target_link_libraries(test_gmp ahnum ${GMP_LIB})

enable_testing()
add_test(NAME gmp_ref_backend COMMAND test_gmp ref)
add_test(NAME gmp_c2p_backend COMMAND test_gmp c2p)
add_test(NAME gmp_neon_backend COMMAND test_gmp neon)
add_test(NAME gmp_neon_threads COMMAND test_gmp neon 40 4)
add_test(NAME gmp_sme_backend COMMAND test_gmp sme 40)
add_test(NAME gmp_sme_threads COMMAND test_gmp sme 30 3)

add_executable(sweep_err tools/sweep_err.c)
target_link_libraries(sweep_err ahnum)

add_executable(ahbench tools/ahbench.c)
target_link_libraries(ahbench ahnum)

add_executable(ahprst tools/ahprst.c)
target_include_directories(ahprst PRIVATE src)
target_link_libraries(ahprst ahnum)

add_executable(test_stages tests/test_stages.c)
target_include_directories(test_stages PRIVATE src)
target_link_libraries(test_stages ahnum)
add_test(NAME asm_stages COMMAND test_stages)
add_test(NAME known_residues COMMAND ${CMAKE_SOURCE_DIR}/tests/run_known.sh $<TARGET_FILE:ahbench> ${CMAKE_SOURCE_DIR}/tests/known_residues.csv 100000)
add_test(NAME known_residues_threads COMMAND ${CMAKE_SOURCE_DIR}/tests/run_known.sh $<TARGET_FILE:ahbench> ${CMAKE_SOURCE_DIR}/tests/known_residues.csv 100000 --threads 2)
add_test(NAME known_residues_long COMMAND ${CMAKE_SOURCE_DIR}/tests/run_known.sh $<TARGET_FILE:ahbench> ${CMAKE_SOURCE_DIR}/tests/known_residues.csv 1000000)
set_tests_properties(known_residues_long PROPERTIES LABELS long)
add_test(NAME prst_compat COMMAND ${CMAKE_SOURCE_DIR}/tests/prst_compat.sh $<TARGET_FILE:ahprst>)
add_test(NAME gerbicz_fault COMMAND $<TARGET_FILE:ahbench> prp 1203 90001 --fault 50000 --expect 4BE9F1953E350085)

add_executable(kbench tools/kbench.c)
target_include_directories(kbench PRIVATE src)
target_link_libraries(kbench ahnum)

add_executable(xcheck tools/xcheck.c)
target_link_libraries(xcheck ahnum)
