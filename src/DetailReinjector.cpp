#include "DetailReinjector.h"
#include <opencv2/imgproc.hpp>

cv::Mat DetailReinjector::Reinject(const cv::Mat& original, const cv::Mat& denoised, const Params& params) {
    cv::Mat residual = original - denoised;

    // Maschera di "dettaglio vero": gradiente locale calcolato sull'immagine
    // GIA' denoised (così il rumore residuo non gonfia artificialmente la maschera),
    // normalizzato e passato attraverso una sigmoide morbida per evitare
    // transizioni brusche che si vedrebbero come un halo.
    cv::Mat gx, gy, gradMag;
    cv::Sobel(denoised, gx, CV_32F, 1, 0, 3);
    cv::Sobel(denoised, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, gradMag);

    cv::Mat mask;
    cv::threshold(gradMag, mask, params.gradientThreshold, 1.0, cv::THRESH_BINARY);
    cv::GaussianBlur(mask, mask, cv::Size(5, 5), 1.5); // ammorbidisce i bordi della maschera stessa

    cv::Mat reinjected = denoised + residual.mul(mask) * params.strength;

    cv::Mat clamped;
    cv::min(reinjected, 1.0, clamped);
    cv::max(clamped, 0.0, clamped);
    return clamped;
}
