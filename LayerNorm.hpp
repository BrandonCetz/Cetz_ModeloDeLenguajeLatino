#pragma once
#include <vector>
#include <cmath>
#include "matrix.hpp"

class LayerNorm {
private:
    size_t dim_;
    float eps_;

    // Caché para Backward Pass analítico exacto
    Matriz X_hat_cache;
    std::vector<float> inv_std_cache;

public:
    LayerNorm(size_t dim, float eps = 1e-5f) : dim_(dim), eps_(eps) {}
    LayerNorm() : dim_(0), eps_(1e-5f) {}

    // Normaliza cada columna (vector de representación del token) de la matriz X
    Matriz forward(const Matriz& X) {
        size_t filas = X.filas();
        size_t cols = X.columnas();
        Matriz Y(filas, cols);
        X_hat_cache = Matriz(filas, cols);
        inv_std_cache.resize(cols);

        for (size_t col = 0; col < cols; ++col) {
            // 1. Calcular Media
            float media = 0.0f;
            for (size_t fila = 0; fila < filas; ++fila) {
                media += X(fila, col);
            }
            media /= static_cast<float>(filas);

            // 2. Calcular Varianza
            float varianza = 0.0f;
            for (size_t fila = 0; fila < filas; ++fila) {
                float diff = X(fila, col) - media;
                varianza += diff * diff;
            }
            varianza /= static_cast<float>(filas);

            // 3. Normalizar
            float inv_std = 1.0f / std::sqrt(varianza + eps_);
            inv_std_cache[col] = inv_std;

            for (size_t fila = 0; fila < filas; ++fila) {
                float x_hat = (X(fila, col) - media) * inv_std;
                X_hat_cache(fila, col) = x_hat;
                Y(fila, col) = x_hat;
            }
        }
        return Y;
    }

    // Derivada analítica exacta de Layer Normalization
    Matriz backward(const Matriz& gradSalida) {
        size_t filas = gradSalida.filas();
        size_t cols = gradSalida.columnas();
        Matriz dX(filas, cols);

        float D = static_cast<float>(filas);

        for (size_t col = 0; col < cols; ++col) {
            float inv_std = (col < inv_std_cache.size()) ? inv_std_cache[col] : 1.0f;

            float sum_g = 0.0f;
            float sum_g_xhat = 0.0f;

            for (size_t fila = 0; fila < filas; ++fila) {
                float g = gradSalida(fila, col);
                float x_hat = X_hat_cache(fila, col);
                sum_g += g;
                sum_g_xhat += g * x_hat;
            }

            float mean_g = sum_g / D;
            float mean_g_xhat = sum_g_xhat / D;

            for (size_t fila = 0; fila < filas; ++fila) {
                float g = gradSalida(fila, col);
                float x_hat = X_hat_cache(fila, col);
                dX(fila, col) = inv_std * (g - mean_g - x_hat * mean_g_xhat);
            }
        }
        return dX;
    }
};
