#pragma once
#include <opencv2/core.hpp>

// Denoise dei canali di CROMINANZA (Cb, Cr) tramite joint/cross bilateral filter
// guidato dagli edge di LUMINANZA.
//
// Motivazione: il rumore di croma è a bassa frequenza spaziale ("blotches" di
// colore), l'occhio umano ha risoluzione cromatica molto più bassa di quella
// di luminanza -> possiamo filtrare la crominanza MOLTO più aggressivamente
// (raggio spaziale ampio) senza perdita percepita di dettaglio, a patto di
// non farla "sanguinare" oltre i bordi reali dell'immagine. Per questo il
// filtro non guarda solo Cb/Cr, ma anche quanto simile è il pixel vicino in
// LUMINANZA (guida): un edge netto in luminanza blocca la diffusione del
// colore attraverso di esso, evitando color bleeding.
namespace BilateralChromaDenoiser {

    struct Params {
        double spatialSigma = 12.0;  // raggio spaziale (px) - più alto = più aggressivo sulla croma
        double chromaSigma  = 25.0;  // tolleranza colore
        double lumaGuideSigma = 8.0; // quanto la differenza di luminanza "blocca" la diffusione
    };

    // channel: singolo canale Cb o Cr (CV_32F). lumaGuide: canale Y (CV_32F), stessa dimensione.
    cv::Mat Denoise(const cv::Mat& channel, const cv::Mat& lumaGuide, const Params& params);
}
