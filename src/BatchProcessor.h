#pragma once
#include "DenoisePipeline.h"
#include <string>
#include <functional>
#include <atomic>

// Elabora un'intera cartella di RAW/JPG in modo che:
//  - non si superi mai un budget di RAM configurabile (fondamentale con
//    centinaia di RAW: un singolo buffer float32 di un sensore 45MP occupa
//    ~540 MB, quindi tenerne 50 in memoria contemporaneamente = 27 GB);
//  - la CPU/GPU restino sempre sature (nessun thread in idle in attesa di I/O);
//  - il progresso sia riportabile in modo thread-safe verso una UI.
//
// Strategia: due pool separati.
//   ioPool_      -> decodifica RAW/JPG da disco (I/O-bound, pochi thread bastano)
//   computePool_ -> esegue DenoisePipeline::Process (CPU/SIMD/GPU-bound, hardware_concurrency())
// Un semaforo (contatore atomico + condition_variable) limita quante
// ImageBuffer decodificate possono stare "in volo" contemporaneamente,
// indipendentemente da quanti file restano da elaborare.
class BatchProcessor {
public:
    struct Options {
        size_t ioThreads = 4;
        size_t computeThreads = std::thread::hardware_concurrency();
        size_t maxRamBudgetMB = 4096; // budget RAM dedicato ai buffer immagine in volo
        std::string outputExtension = ".tif"; // TIFF 16-bit per non perdere qualità in output
    };

    using ProgressCallback = std::function<void(size_t completed, size_t total, const std::string& lastFile)>;

    BatchProcessor(Options options, DenoiseParams denoiseParams);

    // Elabora tutti i file immagine trovati in inputDir, scrive in outputDir.
    // Bloccante: ritorna quando l'intero batch è completato. Per uso da UI,
    // chiamare da un thread dedicato e usare onProgress per aggiornare la barra.
    void ProcessFolder(const std::string& inputDir, const std::string& outputDir, ProgressCallback onProgress = nullptr);

    void RequestCancel() { cancelRequested_.store(true); }

private:
    size_t EstimateInFlightBudget(size_t approxBytesPerImage) const;

    Options options_;
    DenoiseParams denoiseParams_;
    std::atomic<bool> cancelRequested_{false};
};
