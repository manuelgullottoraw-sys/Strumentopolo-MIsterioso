#include "NoiseEstimator.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <vector>

namespace {
    // Un singolo passo di Haar wavelet (analisi) per estrarre la sottobanda HH
    // (dettaglio diagonale), che nella grande maggioranza dei casi contiene
    // quasi solo rumore ad alta frequenza + un po' di micro-texture.
    cv::Mat HaarHighFreqSubband(const cv::Mat& channel) {
        cv::Mat rowsLow, rowsHigh;
        cv::Mat kernelLow  = (cv::Mat_<float>(1, 2) << 0.70710678f, 0.70710678f);
        cv::Mat kernelHigh = (cv::Mat_<float>(1, 2) << 0.70710678f, -0.70710678f);

        cv::Mat tmpLow, tmpHigh;
        cv::filter2D(channel, tmpLow,  CV_32F, kernelLow,  cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
        cv::filter2D(channel, tmpHigh, CV_32F, kernelHigh, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);

        cv::Mat hh;
        cv::Mat kernelLowT = kernelLow.t();
        cv::Mat kernelHighT = kernelHigh.t();
        cv::filter2D(tmpHigh, hh, CV_32F, kernelHighT, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
        return hh;
    }
}

float NoiseEstimator::EstimateSigmaMAD(const cv::Mat& singleChannelFloat) {
    cv::Mat hh = HaarHighFreqSubband(singleChannelFloat);

    std::vector<float> absVals;
    absVals.reserve(hh.total());
    for (int y = 0; y < hh.rows; ++y) {
        const float* row = hh.ptr<float>(y);
        for (int x = 0; x < hh.cols; ++x) {
            absVals.push_back(std::fabs(row[x]));
        }
    }
    if (absVals.empty()) return 0.f;

    std::nth_element(absVals.begin(), absVals.begin() + absVals.size() / 2, absVals.end());
    float median = absVals[absVals.size() / 2];
    return median / 0.6745f;
}

NoiseEstimator::NoiseProfile NoiseEstimator::EstimateProfile(const cv::Mat& ycbcrFloat) {
    std::vector<cv::Mat> planes;
    cv::split(ycbcrFloat, planes); // 0=Y, 1=Cb, 2=Cr

    NoiseProfile profile;
    profile.sigmaLuma     = EstimateSigmaMAD(planes[0]);
    profile.sigmaChromaCb = EstimateSigmaMAD(planes[1]);
    profile.sigmaChromaCr = EstimateSigmaMAD(planes[2]);
    return profile;
}
