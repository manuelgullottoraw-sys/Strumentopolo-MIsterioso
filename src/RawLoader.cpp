#include "RawLoader.h"
#include <libraw/libraw.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <stdexcept>

namespace {
    std::string ToLower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
        return s;
    }

    bool IsRawExtension(const std::string& ext) {
        static const std::vector<std::string> rawExts = {
            ".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2", ".dng", ".orf", ".pef"
        };
        return std::find(rawExts.begin(), rawExts.end(), ext) != rawExts.end();
    }
}

ImageBuffer RawLoader::Load(const std::string& path) {
    std::string ext = ToLower(path.substr(path.find_last_of('.')));
    if (IsRawExtension(ext)) {
        return LoadRawFile(path);
    }
    return LoadStandardImage(path);
}

ImageBuffer RawLoader::LoadRawFile(const std::string& path) {
    LibRaw processor;

    if (processor.open_file(path.c_str()) != LIBRAW_SUCCESS) {
        throw std::runtime_error("LibRaw: impossibile aprire " + path);
    }
    if (processor.unpack() != LIBRAW_SUCCESS) {
        throw std::runtime_error("LibRaw: impossibile decodificare " + path);
    }

    // Parametri di processing: demosaic AHD (qualità alta), niente auto-bright
    // (vogliamo il dato lineare più "grezzo" possibile: la nostra pipeline gestisce
    // tone/denoise da sola), output a 16 bit per canale.
    processor.imgdata.params.output_bps = 16;
    processor.imgdata.params.user_qual  = 3;      // 3 = AHD demosaic
    processor.imgdata.params.no_auto_bright = 1;
    processor.imgdata.params.use_camera_wb = 1;   // white balance dal metadato camera
    processor.imgdata.params.output_color = 1;    // sRGB (linear-light nel nostro buffer dopo conversione)

    if (processor.dcraw_process() != LIBRAW_SUCCESS) {
        throw std::runtime_error("LibRaw: demosaicing fallito per " + path);
    }

    int errCode = 0;
    libraw_processed_image_t* img = processor.dcraw_make_mem_image(&errCode);
    if (!img || errCode != 0) {
        throw std::runtime_error("LibRaw: creazione immagine in memoria fallita per " + path);
    }

    // img->data e' RGB16 interleaved. Lo avvolgiamo in una Mat CV_16UC3 e
    // convertiamo a float32 normalizzato [0,1] in "gamma sRGB" -> poi a lineare.
    cv::Mat raw16(img->height, img->width, CV_16UC3, img->data);
    cv::Mat bgr16;
    cv::cvtColor(raw16, bgr16, cv::COLOR_RGB2BGR);

    cv::Mat asFloat;
    bgr16.convertTo(asFloat, CV_32FC3, 1.0 / 65535.0);

    // Linearizzazione (approssimazione gamma 2.2; per un pipeline color-managed
    // rigoroso sostituire con la EOTF sRGB esatta o LittleCMS + profilo camera).
    cv::Mat linear;
    cv::pow(asFloat, 2.2, linear);

    ImageBuffer::Metadata meta;
    meta.isRawSource  = true;
    meta.isoSpeed     = static_cast<int>(processor.imgdata.other.iso_speed);
    meta.shutterSpeed = processor.imgdata.other.shutter;
    meta.cameraModel  = processor.imgdata.idata.model;

    processor.dcraw_clear_mem(img);
    processor.recycle();

    return ImageBuffer::FromMat(std::move(linear), std::move(meta));
}

ImageBuffer RawLoader::LoadStandardImage(const std::string& path) {
    // IMREAD_ANYDEPTH+IMREAD_COLOR preserva 16-bit se il JPG/TIFF di origine lo fosse
    cv::Mat img = cv::imread(path, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
    if (img.empty()) {
        throw std::runtime_error("OpenCV: impossibile leggere " + path);
    }

    cv::Mat asFloat;
    double scale = (img.depth() == CV_16U) ? 1.0 / 65535.0 : 1.0 / 255.0;
    img.convertTo(asFloat, CV_32FC3, scale);

    cv::Mat linear;
    cv::pow(asFloat, 2.2, linear); // stessa linearizzazione approssimata del path RAW

    ImageBuffer::Metadata meta;
    meta.isRawSource = false;
    return ImageBuffer::FromMat(std::move(linear), std::move(meta));
}
