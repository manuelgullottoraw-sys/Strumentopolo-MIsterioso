#pragma once
#include <opencv2/core.hpp>

// Stima robusta della sigma di rumore per canale, usata per calibrare
// automaticamente la forza di ogni stadio della pipeline (wavelet threshold,
// raggio bilaterale, ricerca NLM) in funzione del rumore reale dello scatto,
// non di uno slider fisso "a occhio".
namespace NoiseEstimator {

    // Stima MAD (median absolute deviation) sui coefficienti di sottobanda
    // ad alta frequenza di una singola decomposizione wavelet (formula di Donoho):
    //   sigma = median(|HH|) / 0.6745
    // Robusta perché la mediana ignora gli outlier (= i bordi/dettagli veri),
    // quindi stima il rumore "puro" senza confonderlo con la struttura dell'immagine.
    float EstimateSigmaMAD(const cv::Mat& singleChannelFloat);

    // Stima sigma separata per luminanza e per ciascun canale di crominanza,
    // opzionalmente combinata con una curva di calibrazione per-ISO/per-camera
    // (equivalente ai "camera noise profile" di Lightroom/DxO).
    struct NoiseProfile {
        float sigmaLuma;
        float sigmaChromaCb;
        float sigmaChromaCr;
    };

    NoiseProfile EstimateProfile(const cv::Mat& ycbcrFloat);
}
