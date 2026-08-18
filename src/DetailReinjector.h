#pragma once
#include <opencv2/core.hpp>

// Ultimo stadio anti-"plastica"/anti-"acquerello": ricalcola quanto dettaglio
// è stato tolto (residuo = originale - denoised) e ne reiniette una frazione
// SOLO dove il residuo correla con struttura reale (bordi/texture ad alto
// gradiente locale), non dove è rumore puro (zone piatte a basso gradiente).
// Questo è concettualmente ciò che separa un buon denoiser classico da un
// semplice blur: il blur toglie E non restituisce mai nulla.
namespace DetailReinjector {

    struct Params {
        float strength = 0.35f;     // 0 = nessuna reiniezione, 1 = massima
        float gradientThreshold = 0.02f; // sopra questa soglia di gradiente locale si considera "dettaglio vero"
    };

    // original e denoised: canale di luminanza (CV_32F), stessa dimensione.
    cv::Mat Reinject(const cv::Mat& original, const cv::Mat& denoised, const Params& params);
}
