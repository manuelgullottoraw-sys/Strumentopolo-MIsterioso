# RawDenoiseBatch — scheletro architetturale

Scheletro C++20 per un denoiser RAW/JPG classico (no reti neurali), pensato
per batch massivi su Windows. Accompagna la risposta discorsiva su
architettura, algoritmi, gestione RAW e ottimizzazione.

## Build automatica su GitHub (CI)

Il repository include `.github/workflows/build.yml`: ogni push su `main` (o
avvio manuale dal tab "Actions") fa compilare il progetto a GitHub su una
macchina Windows, installando da solo OpenCV/LibRaw/OpenCL tramite il
manifest `vcpkg.json`. Il programma compilato (`RawDenoiseBatch.exe`) resta
scaricabile come "artifact" al termine della compilazione.

Attenzione: la primissima compilazione impiega tipicamente **30-60 minuti**,
perché OpenCV con i moduli contrib viene ricompilato da zero da vcpkg. Le
esecuzioni successive sono molto più rapide grazie alla cache configurata nel
workflow.

## Build in locale (Windows, vcpkg)

```
git clone https://github.com/microsoft/vcpkg
.\vcpkg\bootstrap-vcpkg.bat
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=.\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build build --config Release
```

Le dipendenze (OpenCV con modulo `contrib`, LibRaw, OpenCL) sono elencate nel
manifest `vcpkg.json` e vengono scaricate/compilate automaticamente da CMake
in fase di configure: non serve eseguire `vcpkg install` a mano.

## Uso

```
RawDenoiseBatch.exe <cartella_input> <cartella_output> [num_thread_compute]
```

## Struttura

- `RawLoader` — decoding RAW (LibRaw, demosaic AHD) e JPG/TIFF (OpenCV), normalizzati a `ImageBuffer` float32 lineare.
- `NoiseEstimator` — stima robusta (MAD su sottobanda wavelet) della sigma di rumore per luma/croma, per calibrare automaticamente ogni stadio.
- `WaveletDenoiser` — denoise luminanza multi-scala con soft-threshold adattivo (stile BayesShrink) su trasformata undecimated (SWT), preserva i bordi.
- `BilateralChromaDenoiser` — denoise crominanza con joint bilateral filter guidato dagli edge di luminanza, evita color bleeding.
- `NlmLuminanceDenoiser` — passata opzionale Non-Local Means per recuperare micro-texture; percorso CPU (OpenCV, SIMD interno) e hook GPU OpenCL (`src/kernels/nlm_patch_distance.cl`).
- `DetailReinjector` — reiniezione mascherata del residuo (originale - denoised) sulle zone ad alto gradiente, per evitare l'effetto "plastica/acquerello".
- `DenoisePipeline` — orchestratore RGB→YCbCr→[denoise separato]→reiniezione→RGB.
- `ThreadPool` / `BatchProcessor` — due pool separati (I/O e compute), semaforo per limitare i buffer immagine "in volo" e rispettare un budget di RAM configurabile anche con centinaia di file.

## Cosa manca deliberatamente (da completare in produzione)

- Implementazione completa del kernel OpenCL (`DenoiseGpu` è uno stub che ritorna `false` e fa fallback su CPU).
- Trasformata wavelet di sintesi esatta (qui semplificata per leggibilità); per qualità da produzione sostituire con CDF 9/7 o Daubechies e ricostruzione esatta.
- Color management rigoroso (profili ICC via LittleCMS, matrici camera-specifiche) al posto della linearizzazione gamma 2.2 approssimata.
- Persistenza dei parametri utente e binding verso la UI (WinUI 3/WPF, vedi sezione 5 della risposta).
