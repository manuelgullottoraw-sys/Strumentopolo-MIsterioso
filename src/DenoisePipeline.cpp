#include "DenoisePipeline.h"
#include "NoiseEstimator.h"
#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>

ImageBuffer DenoisePipeline::Process(const ImageBuffer& input, const DenoiseParams& params) {
    // 1) RGB lineare -> YCbCr. NB: usare una vera trasformata YCbCr (o CIELAB per
    //    fedeltà percettiva superiore) e non semplicemente HSV: vogliamo un canale
    //    di luminanza che sia effettivamente decorrelato dalla crominanza.
    cv::Mat ycbcr;
    cv::cvtColor(input.data, ycbcr, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> planes; // 0=Y, 1=Cr, 2=Cb (convenzione OpenCV)
    cv::split(ycbcr, planes);
    cv::Mat& luma = planes[0];
    cv::Mat& cr   = planes[1];
    cv::Mat& cb   = planes[2];

    // 2) Stima di rumore robusta e per-canale (adatta la forza dei filtri allo
    //    scatto reale: alto ISO -> soglie più aggressive, basso ISO -> filtri
    //    quasi trasparenti, per non degradare inutilmente immagini già pulite).
    NoiseEstimator::NoiseProfile profile = NoiseEstimator::EstimateProfile(ycbcr);

    // 3) Croma: bilaterale guidato dalla luminanza ORIGINALE (non ancora denoised),
    //    che offre ancora edge affidabili per bloccare il color bleeding.
    cv::Mat crDenoised = BilateralChromaDenoiser::Denoise(cr, luma, params.chroma);
    cv::Mat cbDenoised = BilateralChromaDenoiser::Denoise(cb, luma, params.chroma);

    // 4) Luminanza: passata base wavelet (rimuove il grosso del rumore a tutte
    //    le scale preservando i bordi), poi eventuale passata NLM per recuperare
    //    micro-texture nelle zone ad alta ripetitività (pelle, tessuti, fogliame).
    cv::Mat lumaWavelet = WaveletDenoiser::Denoise(luma, profile.sigmaLuma, params.wavelet);

    cv::Mat lumaDenoised = lumaWavelet;
    if (params.enableNlmPass) {
        NlmLuminanceDenoiser::Params nlmParams = params.nlm;
        nlmParams.strength = std::max(1.0f, profile.sigmaLuma * 40.0f); // calibra "h" sulla sigma reale
        lumaDenoised = NlmLuminanceDenoiser::Denoise(lumaWavelet, nlmParams);
    }

    // 5) Reiniezione dettaglio: confronta la luminanza ORIGINALE con quella
    //    denoised per recuperare la texture vera persa nel processo, mascherata
    //    dal gradiente locale per non riportare dentro anche il rumore.
    cv::Mat lumaFinal = DetailReinjector::Reinject(luma, lumaDenoised, params.detail);

    // 6) Ricomposizione YCbCr -> RGB
    std::vector<cv::Mat> outPlanes = { lumaFinal, crDenoised, cbDenoised };
    cv::Mat ycbcrOut;
    cv::merge(outPlanes, ycbcrOut);

    cv::Mat rgbOut;
    cv::cvtColor(ycbcrOut, rgbOut, cv::COLOR_YCrCb2BGR);

    return ImageBuffer::FromMat(std::move(rgbOut), input.meta);
}
