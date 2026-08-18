// Kernel OpenCL di riferimento: calcola per ogni pixel la distanza patch-patch
// pesata rispetto a tutti i pixel nella finestra di ricerca locale e produce
// il valore denoised come media pesata (Non-Local Means classico).
//
// Ogni work-item gestisce UN pixel di output: la ricerca nella finestra
// (searchRadius) e il confronto patch (patchRadius) sono interamente paralleli
// su GPU, che è dove NLM guadagna più rispetto alla CPU (è la stessa ragione
// per cui BM3D/NLM vengono tipicamente portati su GPU nei tool professionali).
__kernel void nlm_denoise(
    __global const float* input,
    __global float* output,
    const int width,
    const int height,
    const int patchRadius,
    const int searchRadius,
    const float h // parametro di forza del filtro (tolleranza di similarità)
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float weightSum = 0.0f;
    float valueSum  = 0.0f;
    float hh = h * h;

    for (int dy = -searchRadius; dy <= searchRadius; ++dy) {
        for (int dx = -searchRadius; dx <= searchRadius; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;

            // Distanza tra la patch centrata in (x,y) e quella centrata in (nx,ny)
            float patchDist = 0.0f;
            int sampleCount = 0;
            for (int py = -patchRadius; py <= patchRadius; ++py) {
                for (int px = -patchRadius; px <= patchRadius; ++px) {
                    int ax = x + px, ay = y + py;
                    int bx = nx + px, by = ny + py;
                    if (ax < 0 || ay < 0 || ax >= width || ay >= height) continue;
                    if (bx < 0 || by < 0 || bx >= width || by >= height) continue;
                    float diff = input[ay * width + ax] - input[by * width + bx];
                    patchDist += diff * diff;
                    sampleCount++;
                }
            }
            if (sampleCount == 0) continue;
            patchDist /= (float)sampleCount;

            float weight = exp(-patchDist / hh);
            weightSum += weight;
            valueSum  += weight * input[ny * width + nx];
        }
    }

    output[y * width + x] = (weightSum > 0.0f) ? (valueSum / weightSum) : input[y * width + x];
}
