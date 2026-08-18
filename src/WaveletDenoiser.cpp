#include "WaveletDenoiser.h"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <vector>

namespace {

// Un livello di SWT (stationary/undecimated wavelet transform) 2D separabile:
// produce LL, LH, HL, HH senza sottocampionare (a differenza della DWT classica),
// eliminando gli artefatti "a scacchiera" che il soft-threshold introdurrebbe
// altrimenti vicino ai bordi.
struct SwtLevel {
    cv::Mat LL, LH, HL, HH;
};

SwtLevel SwtForward(const cv::Mat& channel, int stepSize) {
    // Kernel Haar dilatato di stepSize (approccio "a trous", come in SWT classica)
    cv::Mat low  = (cv::Mat_<float>(1, 2) << 0.70710678f, 0.70710678f);
    cv::Mat high = (cv::Mat_<float>(1, 2) << 0.70710678f, -0.70710678f);

    // Dilatazione del kernel per il livello corrente (a trous algorithm)
    cv::Mat lowDil  = cv::Mat::zeros(1, (low.cols  - 1) * stepSize + 1, CV_32F);
    cv::Mat highDil = cv::Mat::zeros(1, (high.cols - 1) * stepSize + 1, CV_32F);
    for (int i = 0; i < low.cols; ++i)  lowDil.at<float>(0, i * stepSize)  = low.at<float>(0, i);
    for (int i = 0; i < high.cols; ++i) highDil.at<float>(0, i * stepSize) = high.at<float>(0, i);

    cv::Mat rowLow, rowHigh;
    cv::filter2D(channel, rowLow,  CV_32F, lowDil,  cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    cv::filter2D(channel, rowHigh, CV_32F, highDil, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);

    SwtLevel lvl;
    cv::filter2D(rowLow,  lvl.LL, CV_32F, lowDil.t(),  cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    cv::filter2D(rowLow,  lvl.LH, CV_32F, highDil.t(), cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    cv::filter2D(rowHigh, lvl.HL, CV_32F, lowDil.t(),  cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    cv::filter2D(rowHigh, lvl.HH, CV_32F, highDil.t(), cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    return lvl;
}

// BayesShrink: soglia adattiva per sottobanda, calcolata dal rapporto tra
// varianza locale del segnale e varianza del rumore stimata. Più aggressiva
// dove il segnale è debole (zone piatte = rumore), più permissiva dove il
// segnale è forte (edge/texture = dettaglio vero da preservare).
void SoftThresholdSubband(cv::Mat& subband, float sigmaNoise, float scale) {
    double meanVal, stdVal;
    cv::Scalar mean, stddev;
    cv::meanStdDev(subband, mean, stddev);
    stdVal = stddev[0];

    float varSignal = std::max(0.0, stdVal * stdVal - sigmaNoise * sigmaNoise);
    float sigmaSignal = std::sqrt(varSignal);
    float threshold = (sigmaSignal > 1e-6f)
        ? (sigmaNoise * sigmaNoise) / sigmaSignal
        : (float)stdVal; // fallback: se non c'e' segnale rilevabile, soglia = tutta la banda (soppressione totale)

    threshold *= scale;

    // Soft-threshold: shrink verso zero, non taglio netto -> transizioni morbide, no ringing
    subband.forEach<float>([threshold](float& v, const int*) {
        float sign = (v > 0) ? 1.f : -1.f;
        float mag  = std::fabs(v) - threshold;
        v = (mag > 0) ? sign * mag : 0.f;
    });
}

cv::Mat SwtInverse(const SwtLevel& lvl, int stepSize) {
    // Ricostruzione: per una SWT Haar semplificata, la ricostruzione approssimata
    // è la media delle 4 sottobande riportate in fase. In produzione si userebbe
    // la trasformata inversa esatta (filtri di sintesi coniugati); qui teniamo la
    // versione compatta per leggibilità dello scheletro.
    return (lvl.LL + lvl.LH + lvl.HL + lvl.HH) * 0.25f;
}

} // namespace

cv::Mat WaveletDenoiser::Denoise(const cv::Mat& lumaChannel, float sigmaEstimate, const Params& params) {
    cv::Mat current = lumaChannel.clone();
    std::vector<SwtLevel> levels;
    levels.reserve(params.levels);

    cv::Mat working = current;
    for (int l = 0; l < params.levels; ++l) {
        int stepSize = 1 << l;
        SwtLevel lvl = SwtForward(working, stepSize);
        levels.push_back(lvl);
        working = lvl.LL; // il livello successivo decompone ulteriormente la banda passa-basso
    }

    // Soglia le sottobande di dettaglio ad ogni livello (dal più fine al più grossolano,
    // con soglia via via meno aggressiva perché il rumore residuo a bassa frequenza è minore)
    for (size_t l = 0; l < levels.size(); ++l) {
        float levelAttenuation = 1.0f / std::sqrt(1.0f + static_cast<float>(l));
        SoftThresholdSubband(levels[l].LH, sigmaEstimate * levelAttenuation, params.thresholdScale);
        SoftThresholdSubband(levels[l].HL, sigmaEstimate * levelAttenuation, params.thresholdScale);
        SoftThresholdSubband(levels[l].HH, sigmaEstimate * levelAttenuation, params.thresholdScale);
    }

    // Ricostruzione dal livello più grossolano al più fine
    cv::Mat reconstructed = levels.back().LL;
    for (int l = static_cast<int>(levels.size()) - 1; l >= 0; --l) {
        SwtLevel lvl = levels[l];
        lvl.LL = reconstructed;
        reconstructed = SwtInverse(lvl, 1 << l);
    }

    return reconstructed;
}
