#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <cstdint>

// Buffer immagine di lavoro: SEMPRE float32, 3 canali, range lineare [0,1].
// Lavorare in float32 (anziché 8/16 bit integer) evita banding e clipping
// durante le trasformate multi-stadio (wavelet, bilaterale, reiniezione dettaglio).
struct ImageBuffer {
    cv::Mat data;          // CV_32FC3, working color space (lineare, camera RGB o sRGB-lin)
    int width  = 0;
    int height = 0;

    // Metadati necessari per la stima di rumore adattiva e per i profili per-ISO
    struct Metadata {
        std::string cameraModel;
        int isoSpeed = 0;
        float shutterSpeed = 0.f;
        bool isRawSource = false;
    } meta;

    static ImageBuffer FromMat(cv::Mat m, Metadata metaIn = {}) {
        ImageBuffer buf;
        CV_Assert(m.type() == CV_32FC3);
        buf.data = std::move(m);
        buf.width = buf.data.cols;
        buf.height = buf.data.rows;
        buf.meta = std::move(metaIn);
        return buf;
    }

    size_t ApproxBytes() const {
        return static_cast<size_t>(width) * height * 3 * sizeof(float);
    }
};
