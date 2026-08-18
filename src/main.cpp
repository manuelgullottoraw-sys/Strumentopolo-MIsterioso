#include "BatchProcessor.h"
#include <iostream>
#include <string>

// Harness minimale a riga di comando per validare la pipeline prima di
// collegarla alla UI (WinUI 3 / WPF, vedi sezione 5 della risposta).
// Uso: RawDenoiseBatch <cartella_input> <cartella_output> [thread_compute]
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <input_dir> <output_dir> [compute_threads]\n";
        return 1;
    }

    std::string inputDir  = argv[1];
    std::string outputDir = argv[2];

    BatchProcessor::Options options;
    if (argc >= 4) {
        options.computeThreads = static_cast<size_t>(std::stoi(argv[3]));
    }

    // Parametri di default: valori di partenza ragionevoli, da esporre in UI
    // come slider "Luminance NR", "Chroma NR", "Detail/Sharpness".
    DenoiseParams denoiseParams;
    denoiseParams.wavelet.levels = 4;
    denoiseParams.wavelet.thresholdScale = 1.0f;
    denoiseParams.chroma.spatialSigma = 12.0;
    denoiseParams.chroma.chromaSigma = 25.0;
    denoiseParams.detail.strength = 0.35f;
    denoiseParams.enableNlmPass = true;

    BatchProcessor processor(options, denoiseParams);

    processor.ProcessFolder(inputDir, outputDir, [](size_t done, size_t total, const std::string& lastFile) {
        std::cout << "[" << done << "/" << total << "] completato: " << lastFile << std::endl;
    });

    std::cout << "Batch completato." << std::endl;
    return 0;
}
