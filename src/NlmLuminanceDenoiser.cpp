#include "NlmLuminanceDenoiser.h"
#include <opencv2/photo.hpp>   // cv::fastNlMeansDenoising (CPU, SIMD-ottimizzato internamente da OpenCV)
#include <opencv2/imgproc.hpp>

#ifdef HAVE_OPENCL
#include <CL/cl.h>
#endif

cv::Mat NlmLuminanceDenoiser::Denoise(const cv::Mat& lumaChannel, const Params& params) {
#ifdef HAVE_OPENCL
    if (params.preferGpu) {
        cv::Mat gpuResult;
        if (DenoiseGpu(lumaChannel, params, gpuResult)) {
            return gpuResult;
        }
        // fallthrough al percorso CPU se la GPU non è disponibile o il kernel ha fallito
    }
#endif

    // Percorso CPU: fastNlMeansDenoising di OpenCV usa già SIMD (SSE/AVX via
    // OpenCV universal intrinsics) e multithreading interno tramite il suo
    // TBB/OpenMP backend. Richiede input 8U: convertiamo avanti/indietro
    // mantenendo il buffer principale della pipeline in float32.
    cv::Mat as8u, denoised8u;
    lumaChannel.convertTo(as8u, CV_8U, 255.0);

    cv::fastNlMeansDenoising(
        as8u, denoised8u,
        params.strength,
        params.templateWindowSize,
        params.searchWindowSize
    );

    cv::Mat result;
    denoised8u.convertTo(result, CV_32F, 1.0 / 255.0);
    return result;
}

#ifdef HAVE_OPENCL
bool NlmLuminanceDenoiser::DenoiseGpu(const cv::Mat& lumaChannel, const Params& params, cv::Mat& output) {
    // Scheletro: qui andrebbe l'inizializzazione del contesto OpenCL (cached a
    // livello di processo, non per-immagine!), l'upload del buffer luma,
    // l'esecuzione del kernel in src/kernels/nlm_patch_distance.cl su una
    // griglia 2D (un work-item per pixel, o per patch con riduzione locale),
    // e il download del risultato.
    //
    // Punti chiave di design per l'accelerazione GPU:
    //  - Il contesto/coda di comando OpenCL va creato UNA VOLTA e riutilizzato
    //    per tutte le immagini del batch (creare/distruggere un contesto per
    //    immagine è più lento del guadagno che si otterrebbe).
    //  - Le patch possono essere precaricate in local memory (__local) per
    //    ridurre gli accessi a global memory durante il confronto patch-patch.
    //  - Se il device GPU non supporta i requisiti (es. niente fp32 pieno,
    //    memoria insufficiente per l'immagine), ritornare false qui e lasciare
    //    che il chiamante prosegua sul percorso CPU: il batch non deve mai
    //    bloccarsi per l'assenza di una GPU compatibile.
    (void)lumaChannel; (void)params; (void)output;
    return false; // TODO: implementare il dispatch del kernel OpenCL
}
#endif
