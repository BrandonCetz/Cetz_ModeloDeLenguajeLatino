#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include "matrix.hpp"

using namespace std;

class capaEmbedding {
private:
    Matriz E;          // Matriz de pesos (vocabSize x dimEmbedding)
    Matriz mE, vE;     // Momentos de Adam para la tabla de embeddings

    // Acumulador de gradientes y registro de tokens activos en el batch
    Matriz acum_dE;
    std::vector<bool> tokenActualizado;

    size_t vocabSize_;
    size_t dimEmbedding_;
    size_t ultimoIndice_;

    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;

public:
    capaEmbedding(size_t vocabSize, size_t dimEmbedding)
        : vocabSize_(vocabSize), dimEmbedding_(dimEmbedding), ultimoIndice_(0)
    {
        E = Matriz(vocabSize_, dimEmbedding_);

        mE = Matriz(vocabSize_, dimEmbedding_, 0.0f);
        vE = Matriz(vocabSize_, dimEmbedding_, 0.0f);

        // Inicializar acumuladores
        acum_dE = Matriz(vocabSize_, dimEmbedding_, 0.0f);
        tokenActualizado.assign(vocabSize_, false);

        float limite = std::sqrt(6.0f / (vocabSize_ + dimEmbedding_));
        E.aleatorizar(-limite, limite);
    }

    ~capaEmbedding() {}

    Matriz forward(size_t tokenIndex) {
        if (tokenIndex >= vocabSize_) {
            throw std::out_of_range("Índice de token fuera del tamaño del vocabulario.");
        }

        ultimoIndice_ = tokenIndex;
        Matriz salida(dimEmbedding_, 1);

        for (size_t j = 0; j < dimEmbedding_; ++j) {
            salida(j, 0) = E(tokenIndex, j);
        }

        return salida;
    }

    // Acumula gradientes para el token específico
    void backward(const Matriz& gradienteSalida, size_t tokenIndex) {
        if (tokenIndex >= vocabSize_) return;

        for (size_t j = 0; j < dimEmbedding_; ++j) {
            acum_dE(tokenIndex, j) += gradienteSalida(j, 0);
        }

        // Marcamos que este token participó en el batch actual
        tokenActualizado[tokenIndex] = true;
    }

    // Calcula suma de cuadrados de gradientes acumulados promediados de tokens activos (para Gradient Clipping)
    float calcularNormaGradientesCuadrada(float batchSize) const {
        float suma = 0.0f;
        for (size_t i = 0; i < vocabSize_; ++i) {
            if (!tokenActualizado[i]) continue;
            for (size_t j = 0; j < dimEmbedding_; ++j) {
                float g = acum_dE(i, j) / batchSize;
                suma += g * g;
            }
        }
        return suma;
    }

    // Escala gradientes acumulados de tokens activos por un factor (para Gradient Clipping)
    void escalarGradientesAcumulados(float factor) {
        for (size_t i = 0; i < vocabSize_; ++i) {
            if (!tokenActualizado[i]) continue;
            for (size_t j = 0; j < dimEmbedding_; ++j) {
                acum_dE(i, j) *= factor;
            }
        }
    }

    // Ejecuta Adam SOLO en los tokens que se usaron en el batch
    void aplicarOptimizador(float tasaDeAprendizaje, int t, float batchSize) {
        float corr1 = 1.0f - std::pow(beta1, t);
        float corr2 = 1.0f - std::pow(beta2, t);

        for (size_t i = 0; i < vocabSize_; ++i) {
            // OPTIMIZACIÓN DISPERSA: Si el token no se vio en el batch, lo saltamos.
            if (!tokenActualizado[i]) continue;

            for (size_t j = 0; j < dimEmbedding_; ++j) {
                float g = acum_dE(i, j) / batchSize;

                mE(i, j) = beta1 * mE(i, j) + (1.0f - beta1) * g;
                vE(i, j) = beta2 * vE(i, j) + (1.0f - beta2) * (g * g);

                float m_hat = mE(i, j) / corr1;
                float v_hat = vE(i, j) / corr2;

                E(i, j) -= (tasaDeAprendizaje * m_hat) / (std::sqrt(v_hat) + epsilon);

                acum_dE(i, j) = 0.0f; // Resetear acumulador para este token
            }

            // Apagamos la bandera para el próximo batch
            tokenActualizado[i] = false;
        }
    }

    Matriz obtenerEmbedding(size_t tokenIndex) const {
        Matriz vectorColumna(dimEmbedding_, 1);
        for (size_t j = 0; j < dimEmbedding_; ++j) {
            vectorColumna(j, 0) = E(tokenIndex, j);
        }
        return vectorColumna;
    }

    void guardarBinario(std::ofstream& out) const { E.guardarBinario(out); }
    void cargarBinario(std::ifstream& in) { E.cargarBinario(in); }
};
