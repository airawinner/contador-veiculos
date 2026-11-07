
#include "CImg.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <exception>

namespace fs = std::filesystem;
using namespace cimg_library;

// Verifica se o arquivo possui uma extensão de imagem válida.
bool isImage(const fs::path& path) {
    std::string ext = path.extension().string();

    std::transform(
        ext.begin(),
        ext.end(),
        ext.begin(),
        [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        }
    );

    return ext == ".jpg" ||
           ext == ".jpeg" ||
           ext == ".png";
}

// Converte uma imagem colorida para escala de cinza.
CImg<unsigned char> toGray(
    const CImg<unsigned char>& image
) {
    CImg<unsigned char> gray(
        image.width(),
        image.height(),
        1,
        1,
        0
    );

    const int red = 0;
    const int green = std::min(1, image.spectrum() - 1);
    const int blue = std::min(2, image.spectrum() - 1);

    cimg_forXY(gray, x, y) {
        const float r = image(x, y, 0, red);
        const float g = image(x, y, 0, green);
        const float b = image(x, y, 0, blue);

        const float value =
            0.299f * r +
            0.587f * g +
            0.114f * b;

        gray(x, y) = static_cast<unsigned char>(
            std::clamp(value, 0.0f, 255.0f)
        );
    }

    return gray;
}

// Aplica o operador Sobel para detectar bordas.
CImg<unsigned char> sobel(
    const CImg<unsigned char>& image
) {
    CImg<unsigned char> edges(
        image.width(),
        image.height(),
        1,
        1,
        0
    );

    const int gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    const int gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (int y = 1; y < image.height() - 1; ++y) {
        for (int x = 1; x < image.width() - 1; ++x) {

            float sumX = 0.0f;
            float sumY = 0.0f;

            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {

                    const float pixel =
                        image(x + kx, y + ky);

                    sumX += pixel * gx[ky + 1][kx + 1];
                    sumY += pixel * gy[ky + 1][kx + 1];
                }
            }

            const float magnitude = std::sqrt(
                sumX * sumX + sumY * sumY
            );

            edges(x, y) = static_cast<unsigned char>(
                std::clamp(magnitude / 4.0f, 0.0f, 255.0f)
            );
        }
    }

    return edges;
}

// Equaliza o histograma da imagem.
CImg<unsigned char> equalize(
    const CImg<unsigned char>& image
) {
    std::array<unsigned long, 256> histogram{};

    const unsigned long total =
        static_cast<unsigned long>(image.width()) *
        static_cast<unsigned long>(image.height());

    cimg_forXY(image, x, y) {
        ++histogram[image(x, y)];
    }

    std::array<unsigned long, 256> cdf{};

    cdf[0] = histogram[0];

    for (int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    unsigned long cdfMin = 0;

    for (int i = 0; i < 256; ++i) {
        if (histogram[i] > 0) {
            cdfMin = cdf[i];
            break;
        }
    }

    std::array<unsigned char, 256> lookup{};

    if (total == cdfMin) {
        for (int i = 0; i < 256; ++i) {
            lookup[i] = static_cast<unsigned char>(i);
        }
    } else {
        for (int i = 0; i < 256; ++i) {
            const float value =
                255.0f *
                static_cast<float>(cdf[i] - cdfMin) /
                static_cast<float>(total - cdfMin);

            lookup[i] = static_cast<unsigned char>(
                std::clamp(value, 0.0f, 255.0f)
            );
        }
    }

    CImg<unsigned char> result = image;

    cimg_forXY(result, x, y) {
        result(x, y) = lookup[image(x, y)];
    }

    return result;
}

// Processa uma imagem e salva as quatro versões.
void processImage(
    const fs::path& input,
    const fs::path& output
) {
    try {
        CImg<unsigned char> image(input.string().c_str());

        // 1. Escala de cinza.
        CImg<unsigned char> gray = toGray(image);

        // 2. Filtro Gaussiano.
        CImg<unsigned char> gaussian =
            gray.get_blur(1.2f);

        // 3. Detecção de bordas com Sobel.
        CImg<unsigned char> edges =
            sobel(gaussian);

        // 4. Equalização do histograma.
        CImg<unsigned char> equalized =
            equalize(gray);

        // Cria os diretórios de saída.
        fs::create_directories(output / "grayscale");
        fs::create_directories(output / "gaussian");
        fs::create_directories(output / "sobel");
        fs::create_directories(output / "equalized");

        const std::string filename =
            input.stem().string() + ".png";

        // Salva escala de cinza.
        gray.save_png(
            (output / "grayscale" / filename).string().c_str()
        );

        // Salva filtro Gaussiano.
        gaussian.save_png(
            (output / "gaussian" / filename).string().c_str()
        );

        // Salva detecção de bordas.
        edges.save_png(
            (output / "sobel" / filename).string().c_str()
        );

        // Salva equalização do histograma.
        equalized.save_png(
            (output / "equalized" / filename).string().c_str()
        );

        std::cout << "Processada: "
                  << input.filename().string()
                  << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Erro ao processar "
                  << input.filename().string()
                  << ": " << e.what()
                  << std::endl;
    }
}

// Programa principal.
int main(int argc, char* argv[]) {

    if (argc < 3) {
        std::cerr
            << "Uso: ./cimg_preprocess <entrada> <saida>"
            << std::endl;

        return 1;
    }

    const fs::path input = argv[1];
    const fs::path output = argv[2];

    if (!fs::exists(input)) {
        std::cerr
            << "Erro: caminho de entrada nao encontrado."
            << std::endl;

        return 1;
    }

    std::cout << "Iniciando processamento CImg..."
              << std::endl;

    if (fs::is_regular_file(input)) {

        if (!isImage(input)) {
            std::cerr << "Arquivo nao e uma imagem valida."
                      << std::endl;
            return 1;
        }

        processImage(input, output);

    } else if (fs::is_directory(input)) {

        int processed = 0;

        for (const auto& entry : fs::directory_iterator(input)) {

            if (entry.is_regular_file() &&
                isImage(entry.path())) {

                processImage(entry.path(), output);
                ++processed;
            }
        }

        std::cout << "\nTotal de imagens processadas: "
                  << processed
                  << std::endl;

    } else {
        std::cerr << "Entrada invalida." << std::endl;
        return 1;
    }

    std::cout << "\nProcessamento finalizado!"
              << std::endl;

    return 0;
}
