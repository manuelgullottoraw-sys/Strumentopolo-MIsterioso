#include "BatchProcessor.h"
#include "RawLoader.h"
#include "ThreadPool.h"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <semaphore>
#include <iostream>
#include <vector>
#include <future>

namespace fs = std::filesystem;

BatchProcessor::BatchProcessor(Options options, DenoiseParams denoiseParams)
    : options_(std::move(options)), denoiseParams_(std::move(denoiseParams)) {}

size_t BatchProcessor::EstimateInFlightBudget(size_t approxBytesPerImage) const {
    size_t budgetBytes = options_.maxRamBudgetMB * 1024ull * 1024ull;
    // Fattore 3x: buffer originale + buffer intermedio (YCbCr/planes) + buffer output
    // in volo contemporaneamente per la stessa immagine durante la pipeline.
    size_t perImageWithOverhead = approxBytesPerImage * 3;
    size_t maxInFlight = perImageWithOverhead > 0 ? budgetBytes / perImageWithOverhead : 1;
    return std::max<size_t>(1, maxInFlight);
}

void BatchProcessor::ProcessFolder(const std::string& inputDir, const std::string& outputDir, ProgressCallback onProgress) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.is_regular_file()) files.push_back(entry.path().string());
    }
    if (files.empty()) return;

    fs::create_directories(outputDir);

    // Stima approssimativa della dimensione media immagine (usata solo per
    // dimensionare il semaforo prima di conoscere le dimensioni reali dei file
    // di questo batch specifico; in una versione più raffinata si potrebbe
    // leggere le dimensioni JPEG/RAW header-only per una stima più precisa).
    constexpr size_t assumedBytesPerImage = 500ull * 1024 * 1024; // ~ipotesi 45MP float32
    size_t maxInFlight = EstimateInFlightBudget(assumedBytesPerImage);

    // std::counting_semaphore (C++20): limita quante immagini possono essere
    // "decodificate ma non ancora scritte su disco" in un dato istante,
    // impedendo che il productor (ioPool) superi il consumer (computePool)
    // e saturi la RAM con centinaia di buffer in attesa.
    std::counting_semaphore<> inFlightSlots(static_cast<ptrdiff_t>(maxInFlight));

    ThreadPool ioPool(options_.ioThreads);
    ThreadPool computePool(options_.computeThreads);

    std::atomic<size_t> completed{0};
    std::mutex progressMutex;

    std::vector<std::future<void>> allJobs;
    allJobs.reserve(files.size());

    for (const auto& path : files) {
        if (cancelRequested_.load()) break;

        allJobs.push_back(ioPool.Enqueue([this, path, &outputDir, &computePool, &inFlightSlots,
                                           &completed, &progressMutex, &onProgress, total = files.size()] {
            inFlightSlots.acquire(); // blocca se il budget RAM è saturo: back-pressure naturale

            try {
                ImageBuffer loaded = RawLoader::Load(path);

                // Il denoise (CPU/SIMD/GPU-bound) va sul pool separato, così il
                // pool di I/O può già iniziare a decodificare il file successivo
                // mentre questo viene elaborato: pipeline a stadi sovrapposti,
                // non strettamente sequenziale per-file.
                auto computeFuture = computePool.Enqueue([this, loaded = std::move(loaded), path]() -> ImageBuffer {
                    return DenoisePipeline::Process(loaded, denoiseParams_);
                });

                ImageBuffer result = computeFuture.get();

                fs::path inPath(path);
                fs::path outPath = fs::path(outputDir) / (inPath.stem().string() + options_.outputExtension);

                cv::Mat outputForWrite;
                result.data.convertTo(outputForWrite, CV_16U, 65535.0); // torna a 16-bit integer solo in scrittura
                cv::imwrite(outPath.string(), outputForWrite);

            } catch (const std::exception& ex) {
                std::cerr << "Errore su " << path << ": " << ex.what() << std::endl;
            }

            inFlightSlots.release();

            size_t done = ++completed;
            if (onProgress) {
                std::lock_guard<std::mutex> lock(progressMutex);
                onProgress(done, total, path);
            }
        }));
    }

    for (auto& job : allJobs) job.wait();
}
