#pragma once
#include <opencv2/core.hpp>

// Passata opzionale di Non-Local Means sulla luminanza per recuperare la
// texture fine che il thresholding wavelet, essendo un'operazione più
// "globale" per sottobanda, può appiattire leggermente (pelle, tessuti,
// fogliame). NLM confronta PATCH intere (non singoli pixel) in una finestra
// di ricerca locale e media i pixel di patch simili: preserva la texture
// ripetitiva molto meglio di un filtro puramente spaziale.
//
// Complessità nativa O(N * W^2 * P^2) (N=pixel, W=finestra ricerca, P=patch).
// Qui la teniamo trattabile per il batch massivo con due leve:
//   1) finestra di ricerca limitata (non tutta l'immagine, come nel BM3D classico)
//   2) percorso GPU via OpenCL per il calcolo delle distanze patch-patch,
//      che è l'operazione dominante ed è "embarrassingly parallel".
// Su CPU, la stessa funzione usa OpenCV (già vettorizzato SIMD internamente)
// come fallback quando non è disponibile una GPU compatibile.
namespace NlmLuminanceDenoiser {

    struct Params {
        int   templateWindowSize = 7;  // dimensione patch (dispari)
        int   searchWindowSize   = 21; // finestra di ricerca locale (dispari)
        float strength = 6.0f;         // "h" del filtro NLM, calibrato da sigma stimata
        bool  preferGpu = true;
    };

    cv::Mat Denoise(const cv::Mat& lumaChannel, const Params& params);

#ifdef HAVE_OPENCL
    // Implementazione GPU: calcola le distanze patch-patch nel kernel OpenCL
    // (src/kernels/nlm_patch_distance.cl) e aggrega i pesi su GPU.
    // Ritorna false se nessun device OpenCL idoneo è disponibile, così il
    // chiamante può ricadere sul percorso CPU senza interrompere il batch.
    bool DenoiseGpu(const cv::Mat& lumaChannel, const Params& params, cv::Mat& output);
#endif
}
