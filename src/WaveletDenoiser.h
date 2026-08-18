#pragma once
#include <opencv2/core.hpp>

// Denoise multi-scala del canale di LUMINANZA tramite trasformata wavelet
// (qui: Haar ridondante/undecimated per evitare artefatti a blocchi) e
// soft-thresholding adattivo per sottobanda (stile BayesShrink).
//
// Perché la wavelet e non un semplice blur: i coefficienti wavelet nei punti
// di edge/dettaglio hanno magnitudine alta e SOPRAVVIVONO alla soglia, mentre
// il rumore (a banda larga, magnitudine bassa e uniforme) viene azzerato.
// Risultato: i bordi restano netti, il rumore nelle zone piatte scompare,
// senza il tipico effetto "acquerello" di un gaussian/median blur.
namespace WaveletDenoiser {

    struct Params {
        int   levels = 4;          // livelli di decomposizione (più livelli = rumore a bassa frequenza rimosso meglio)
        float thresholdScale = 1.0f; // moltiplicatore globale (esposto in UI come "Luminance NR")
        bool  useUndecimated = true; // true = niente downsampling -> no artefatti, più costoso in CPU/RAM
    };

    // sigmaEstimate proviene da NoiseEstimator::EstimateProfile (robusto, per-immagine).
    // Ritorna il canale di luminanza denoised, stessa dimensione/tipo dell'input (CV_32F, 1 canale).
    cv::Mat Denoise(const cv::Mat& lumaChannel, float sigmaEstimate, const Params& params);
}
