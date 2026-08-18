#pragma once
#include "ImageBuffer.h"
#include "WaveletDenoiser.h"
#include "BilateralChromaDenoiser.h"
#include "NlmLuminanceDenoiser.h"
#include "DetailReinjector.h"

// Orchestratore: RGB -> YCbCr -> [denoise separato luma/croma] -> reiniezione
// dettaglio -> YCbCr -> RGB. Questa è la sequenza discussa nella sezione 1:
// separare luminanza e crominanza permette forza di filtro molto diversa
// per ciascuna, evitando sia il rumore di colore residuo sia la perdita di
// nitidezza sui dettagli fini in luminanza.
struct DenoiseParams {
    WaveletDenoiser::Params wavelet;
    BilateralChromaDenoiser::Params chroma;
    NlmLuminanceDenoiser::Params nlm;
    DetailReinjector::Params detail;
    bool enableNlmPass = true; // disattivabile per batch molto grandi a bassa potenza di calcolo
};

namespace DenoisePipeline {
    ImageBuffer Process(const ImageBuffer& input, const DenoiseParams& params);
}
