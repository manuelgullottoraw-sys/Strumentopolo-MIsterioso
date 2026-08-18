#include "BilateralChromaDenoiser.h"
#include <opencv2/ximgproc.hpp> // jointBilateralFilter (modulo opencv_contrib)
#include <opencv2/imgproc.hpp>

cv::Mat BilateralChromaDenoiser::Denoise(const cv::Mat& channel, const cv::Mat& lumaGuide, const Params& params) {
    cv::Mat output;

    // jointBilateralFilter: il peso di ogni vicino dipende sia dalla distanza
    // spaziale sia dalla similarità nel canale GUIDA (luminanza), non nel
    // canale filtrato stesso. Questo è ciò che rende la croma "consapevole"
    // dei bordi reali dell'immagine anche se il canale colore da solo è troppo
    // rumoroso per definirli in modo affidabile.
    cv::ximgproc::jointBilateralFilter(
        lumaGuide,
        channel,
        output,
        /*d=*/-1, // calcolato da spatialSigma
        params.chromaSigma,
        params.spatialSigma
    );

    return output;
}
