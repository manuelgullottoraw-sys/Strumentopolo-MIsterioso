cmake_minimum_required(VERSION 3.20)
project(RawDenoiseBatch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Dipendenze esterne
#   - OpenCV (+ moduli contrib "photo" e "ximgproc"): I/O immagini, NLM,
#     bilateral/joint-bilateral filter, algebra su Mat
#   - LibRaw: decoding + demosaicing dei file RAW
#   - (opzionale) OpenCL per l'accelerazione GPU del blocco NLM/patch-matching
# Le dipendenze sono dichiarate nel manifest vcpkg.json nella root del
# progetto: se si compila con il toolchain file di vcpkg (vedi README e il
# workflow GitHub Actions in .github/workflows/build.yml), vengono scaricate
# e compilate automaticamente in fase di configure, senza bisogno di
# installarle a mano.
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs photo ximgproc)
find_package(LibRaw REQUIRED)
find_package(OpenCL QUIET)

if(MSVC)
    # Abilita AVX2 per il codice SIMD scritto a mano e per l'auto-vettorizzazione del compilatore
    add_compile_options(/arch:AVX2 /fp:fast /openmp)
else()
    add_compile_options(-mavx2 -mfma -ffast-math -fopenmp)
endif()

add_executable(RawDenoiseBatch
    src/main.cpp
    src/RawLoader.cpp
    src/NoiseEstimator.cpp
    src/WaveletDenoiser.cpp
    src/BilateralChromaDenoiser.cpp
    src/NlmLuminanceDenoiser.cpp
    src/DetailReinjector.cpp
    src/DenoisePipeline.cpp
    src/BatchProcessor.cpp
)

target_include_directories(RawDenoiseBatch PRIVATE src)
target_link_libraries(RawDenoiseBatch PRIVATE ${OpenCV_LIBS} LibRaw::LibRaw)

if(OpenCL_FOUND)
    target_compile_definitions(RawDenoiseBatch PRIVATE HAVE_OPENCL=1)
    target_link_libraries(RawDenoiseBatch PRIVATE OpenCL::OpenCL)
endif()
