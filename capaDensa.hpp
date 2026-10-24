#pragma once
#include <iostream>
#include <functional>
#include <cmath>
#include "matrix.hpp"

using namespace std;

inline float lineal(float x){
    return x;
}
inline float derivadaLineal(float /*a*/){
    return 1.0f;
}
inline float sigmoide(float x){
    return (1.0f / (1.0f + exp(-x)));
}
inline float derivadaSigmoide(float a){
    return a * (1.0f - a);
}
inline float ReLU(float x){
    return x > 0.0f ? x : 0.0f;
}
inline float derivadaReLU(float a){
    return a > 0.0f ? 1.0f : 0.0f;
}
inline float tangenteHiperbolica(float x){
    return tanh(x);
}
inline float derivadaTanHiper(float x){
    return 1.0f - x * x;
}

class capaDensa {
private:
    bool esSoftmax_ = false;
    Matriz W, B, X, Z, A;
    function<float(float)> activacion_, derivadaActivacion_;

    // Acumuladores de Adam
    Matriz mW, vW;
    Matriz mB, vB;

    // Acumuladores de gradientes para Mini-Batching
    Matriz acum_dW, acum_dB;

    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;

public:
    capaDensa() : esSoftmax_(false) {}
    capaDensa(size_t entradas, size_t salidas,
              function<float(float)> act = sigmoide,
              function<float(float)> dAct = derivadaSigmoide) {

        activacion_ = act;
        derivadaActivacion_ = dAct;

        W = Matriz(salidas, entradas);
        B = Matriz(salidas, 1);

        mW = Matriz(salidas, entradas, 0.0f);
        vW = Matriz(salidas, entradas, 0.0f);
        mB = Matriz(salidas, 1, 0.0f);
        vB = Matriz(salidas, 1, 0.0f);

        // Inicializar acumuladores de gradientes en ceros
        acum_dW = Matriz(salidas, entradas, 0.0f);
        acum_dB = Matriz(salidas, 1, 0.0f);

        float limite = sqrt(6.0f / (entradas + salidas));
        W.aleatorizar(-limite, limite);
        B.aleatorizar(0.0f, 0.0f);
    }

    ~capaDensa() {}

    void usarSoftmax() { esSoftmax_ = true; }

    Matriz forward(const Matriz& X_in) {
        this->X = X_in;
        Z = W.matmul(X);

        for (size_t col = 0; col < Z.columnas(); ++col) {
            for (size_t fila = 0; fila < Z.filas(); ++fila) {
                Z(fila, col) += B(fila, 0);
            }
        }

        if (esSoftmax_) {
            A = Z.softmaxColumnas();
        } else {
            A = Z.aplicar(activacion_);
        }
        return A;
    }

    // Acumula gradientes para Mini-Batching
    Matriz backward(const Matriz& gradienteSalida) {
        Matriz Delta;

        if (esSoftmax_) {
            Delta = gradienteSalida;
        } else {
            Matriz dA = A.aplicar(derivadaActivacion_);
            Delta = gradienteSalida.hadamard(dA);
        }

        Matriz dW = Delta.matmul(X.transpuesta());
        Matriz dX = W.transpuesta().matmul(Delta);

        // Acumular gradientes de Pesos (W)
        acum_dW.sumarInPlace(dW);

        // Acumular gradientes de Sesgos (B) a lo largo del batch de tokens
        for (size_t i = 0; i < B.filas(); ++i) {
            float g = 0.0f;
            for (size_t col = 0; col < Delta.columnas(); ++col) {
                g += Delta(i, col);
            }
            B(i, 0); // Placeholder for logic check
            acum_dB(i, 0) += g;
        }

        return dX;
    }

    // Calcula suma de cuadrados de gradientes acumulados promediados (para Gradient Clipping)
    float calcularNormaGradientesCuadrada(float batchSize) const {
        float suma = 0.0f;
        for (size_t i = 0; i < W.filas(); ++i) {
            for (size_t j = 0; j < W.columnas(); ++j) {
                float g = acum_dW(i, j) / batchSize;
                suma += g * g;
            }
        }
        for (size_t i = 0; i < B.filas(); ++i) {
            float g = acum_dB(i, 0) / batchSize;
            suma += g * g;
        }
        return suma;
    }

    // Escala gradientes acumulados por un factor (para Gradient Clipping)
    void escalarGradientesAcumulados(float factor) {
        for (size_t i = 0; i < W.filas(); ++i) {
            for (size_t j = 0; j < W.columnas(); ++j) {
                acum_dW(i, j) *= factor;
            }
        }
        for (size_t i = 0; i < B.filas(); ++i) {
            acum_dB(i, 0) *= factor;
        }
    }

    // Ejecuta Adam con los gradientes promediados del Batch
    void aplicarOptimizador(float tasaDeAprendizaje, int t, float batchSize) {
        float corr1 = 1.0f - pow(beta1, t);
        float corr2 = 1.0f - pow(beta2, t);

        // 1. Adam para Pesos (W)
        for (size_t i = 0; i < W.filas(); ++i) {
            for (size_t j = 0; j < W.columnas(); ++j) {
                float g = acum_dW(i, j) / batchSize;

                mW(i, j) = beta1 * mW(i, j) + (1.0f - beta1) * g;
                vW(i, j) = beta2 * vW(i, j) + (1.0f - beta2) * (g * g);

                float m_hat = mW(i, j) / corr1;
                float v_hat = vW(i, j) / corr2;

                W(i, j) -= (tasaDeAprendizaje * m_hat) / (sqrt(v_hat) + epsilon);

                acum_dW(i, j) = 0.0f;
            }
        }

        // 2. Adam para Sesgos (B)
        for (size_t i = 0; i < B.filas(); ++i) {
            float g = acum_dB(i, 0) / batchSize;

            mB(i, 0) = beta1 * mB(i, 0) + (1.0f - beta1) * g;
            vB(i, 0) = beta2 * vB(i, 0) + (1.0f - beta2) * (g * g);

            float m_hat = mB(i, 0) / corr1;
            float v_hat = vB(i, 0) / corr2;

            B(i, 0) -= (tasaDeAprendizaje * m_hat) / (sqrt(v_hat) + epsilon);

            acum_dB(i, 0) = 0.0f;
        }
    }

    void guardar(ofstream& out) const {
        W.guardarArchivo(out);
        B.guardarArchivo(out);
    }

    void cargar(ifstream& in) {
        W.cargarArchivo(in);
        B.cargarArchivo(in);
    }
    void guardarBinario(std::ofstream& out) const {
        W.guardarBinario(out);
        B.guardarBinario(out);
    }
    void cargarBinario(std::ifstream& in) {
        W.cargarBinario(in);
        B.cargarBinario(in);
    }
};
