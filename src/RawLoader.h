#pragma once
#include "ImageBuffer.h"
#include <string>

// Carica file RAW (via LibRaw, con demosaicing AHD/DHT a 16 bit lineare) o
// JPG/TIFF (via OpenCV) e li normalizza tutti allo stesso formato di lavoro:
// ImageBuffer float32 lineare 0..1.
class RawLoader {
public:
    // Riconosce l'estensione e instrada al loader corretto.
    static ImageBuffer Load(const std::string& path);

private:
    static ImageBuffer LoadRawFile(const std::string& path);
    static ImageBuffer LoadStandardImage(const std::string& path); // JPG/PNG/TIFF già demosaicati
};
