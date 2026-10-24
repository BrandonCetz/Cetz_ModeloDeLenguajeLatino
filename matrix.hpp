#ifndef MATRIX_HPP
#define MATRIX_HPP
#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <stdexcept>
#include <iomanip>
#include <cmath>
#include <functional>
#include <fstream>
#include <omp.h>

class Matriz {
private:
    size_t filas_;
    size_t columnas_;
    std::vector<float> datos_;

public:
    Matriz() : filas_(0), columnas_(0) {}
    Matriz(size_t filas, size_t columnas, float init_val = 0.0f)
        : filas_(filas), columnas_(columnas), datos_(filas * columnas, init_val) {}

    size_t filas() const { return filas_; }
    size_t columnas() const { return columnas_; }

    float& operator()(size_t r, size_t c) {
        return datos_[r * columnas_ + c];
    }
    const float& operator()(size_t r, size_t c) const {
        return datos_[r * columnas_ + c];
    }

    float* data() { return datos_.data(); }
    const float* data() const { return datos_.data(); }

    void aleatorizar(float min_val = -1.0f, float max_val = 1.0f) {
        static std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dist(min_val, max_val);
        for (float& val : datos_) {
            val = dist(gen);
        }
    }

    Matriz operator+(const Matriz& A) const {
        if (filas_ != A.filas_ || columnas_ != A.columnas_) {
            throw std::invalid_argument("Dimensiones incompatibles en suma (+).");
        }
        Matriz resultado(filas_, columnas_);
        size_t total = datos_.size();
        const float* d1 = datos_.data();
        const float* d2 = A.datos_.data();
        float* res = resultado.datos_.data();

        #pragma omp parallel for if(total >= 16384) schedule(static)
        for (size_t i = 0; i < total; ++i) {
            res[i] = d1[i] + d2[i];
        }
        return resultado;
    }

    Matriz operator-(const Matriz& A) const {
        if (filas_ != A.filas_ || columnas_ != A.columnas_) {
            throw std::invalid_argument("Dimensiones incompatibles en resta (-).");
        }
        Matriz resultado(filas_, columnas_);
        size_t total = datos_.size();
        const float* d1 = datos_.data();
        const float* d2 = A.datos_.data();
        float* res = resultado.datos_.data();

        #pragma omp parallel for if(total >= 16384) schedule(static)
        for (size_t i = 0; i < total; ++i) {
            res[i] = d1[i] - d2[i];
        }
        return resultado;
    }

    Matriz operator*(float A) const {
        Matriz resultado(filas_, columnas_);
        size_t total = datos_.size();
        const float* d1 = datos_.data();
        float* res = resultado.datos_.data();

        #pragma omp parallel for if(total >= 16384) schedule(static)
        for (size_t i = 0; i < total; ++i) {
            res[i] = d1[i] * A;
        }
        return resultado;
    }

    // MULTIPLICACIÓN MATRICIAL PARALELIZADA MULTI-HILO (OpenMP + Vectorización SIMD)
    Matriz matmul(const Matriz& A) const {
        if (columnas_ != A.filas_) {
            throw std::invalid_argument("Dimensiones incompatibles en matmul.");
        }
        Matriz resultado(filas_, A.columnas_, 0.0f);

        const float* ptrA = datos_.data();
        const float* ptrB = A.datos_.data();
        float* ptrRes = resultado.datos_.data();

        size_t M = filas_;
        size_t K = columnas_;
        size_t N = A.columnas_;
        size_t totalOps = M * K * N;

        // Paralelización adaptativa por filas si el volumen de cómputo justifica OpenMP
        if (totalOps >= 8192 && M >= 4) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < M; ++i) {
                size_t iN = i * N;
                size_t iK = i * K;
                for (size_t k = 0; k < K; ++k) {
                    float a_ik = ptrA[iK + k];
                    size_t kN = k * N;
                    #pragma GCC ivdep
                    for (size_t j = 0; j < N; ++j) {
                        ptrRes[iN + j] += a_ik * ptrB[kN + j];
                    }
                }
            }
        } else if (totalOps >= 8192 && M < 4 && N >= 32) {
            // Si M es pequeño (ej. vector de token 1xK) pero N es grande, paralelizamos por columnas
            #pragma omp parallel for schedule(static)
            for (size_t j = 0; j < N; ++j) {
                for (size_t i = 0; i < M; ++i) {
                    float sum = 0.0f;
                    size_t iK = i * K;
                    for (size_t k = 0; k < K; ++k) {
                        sum += ptrA[iK + k] * ptrB[k * N + j];
                    }
                    ptrRes[i * N + j] = sum;
                }
            }
        } else {
            // Secuencial optimizado para matrices pequeñas (cero sobrecarga de hilos)
            for (size_t i = 0; i < M; ++i) {
                size_t iN = i * N;
                size_t iK = i * K;
                for (size_t k = 0; k < K; ++k) {
                    float a_ik = ptrA[iK + k];
                    size_t kN = k * N;
                    for (size_t j = 0; j < N; ++j) {
                        ptrRes[iN + j] += a_ik * ptrB[kN + j];
                    }
                }
            }
        }
        return resultado;
    }

    Matriz transpuesta() const {
        Matriz resultado(columnas_, filas_);
        size_t M = filas_;
        size_t N = columnas_;
        size_t total = M * N;

        #pragma omp parallel for if(total >= 16384) schedule(static)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                resultado(j, i) = (*this)(i, j);
            }
        }
        return resultado;
    }

    Matriz hadamard(const Matriz& A) const {
        if (columnas_ != A.columnas_ || filas_ != A.filas_) {
            throw std::invalid_argument("Dimensiones incompatibles en Hadamard.");
        }
        Matriz resultado(filas_, columnas_, 0.0f);
        size_t total = datos_.size();
        const float* d1 = datos_.data();
        const float* d2 = A.datos_.data();
        float* res = resultado.datos_.data();

        #pragma omp parallel for if(total >= 16384) schedule(static)
        for (size_t i = 0; i < total; ++i) {
            res[i] = d1[i] * d2[i];
        }
        return resultado;
    }

    Matriz aplicar(std::function<float(float)> func) const {
        Matriz resultado(filas_, columnas_);
        for (size_t i = 0; i < datos_.size(); ++i) {
            resultado.datos_[i] = func(datos_[i]);
        }
        return resultado;
    }

    Matriz softmax() const {
        Matriz resultado(filas_, columnas_);
        if (datos_.empty()) return resultado;

        float max_val = datos_[0];
        for (size_t i = 1; i < datos_.size(); ++i) {
            if (datos_[i] > max_val) max_val = datos_[i];
        }

        float suma_exp = 0.0f;
        for (size_t i = 0; i < datos_.size(); ++i) {
            resultado.datos_[i] = std::exp(datos_[i] - max_val);
            suma_exp += resultado.datos_[i];
        }

        float inv_suma = 1.0f / (suma_exp > 1e-12f ? suma_exp : 1e-12f);
        for (size_t i = 0; i < datos_.size(); ++i) {
            resultado.datos_[i] *= inv_suma;
        }
        return resultado;
    }

    // Softmax aplicado columna por columna (para secuencias de tokens en LM Head)
    Matriz softmaxColumnas() const {
        Matriz resultado(filas_, columnas_);
        if (datos_.empty()) return resultado;
        size_t F = filas_;
        size_t C = columnas_;

        #pragma omp parallel for if(C >= 4) schedule(static)
        for (size_t col = 0; col < C; ++col) {
            float max_val = (*this)(0, col);
            for (size_t fila = 1; fila < F; ++fila) {
                float v = (*this)(fila, col);
                if (v > max_val) max_val = v;
            }

            float suma_exp = 0.0f;
            for (size_t fila = 0; fila < F; ++fila) {
                float e = std::exp((*this)(fila, col) - max_val);
                resultado(fila, col) = e;
                suma_exp += e;
            }

            float inv_suma = 1.0f / (suma_exp > 1e-12f ? suma_exp : 1e-12f);
            for (size_t fila = 0; fila < F; ++fila) {
                resultado(fila, col) *= inv_suma;
            }
        }
        return resultado;
    }

    inline void rellenarCero() {
        std::fill(datos_.begin(), datos_.end(), 0.0f);
    }

    inline void copiarDe(const Matriz& otra) {
        if (filas_ == otra.filas_ && columnas_ == otra.columnas_) {
            std::copy(otra.datos_.begin(), otra.datos_.end(), datos_.begin());
        } else {
            *this = otra;
        }
    }

    void imprimir(const std::string& name = "Matriz") const {
        std::cout << name << " (" << filas_ << "x" << columnas_ << "):\n";
        for (size_t i = 0; i < filas_; ++i) {
            std::cout << "[";
            for (size_t j = 0; j < columnas_; ++j) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(4) << (*this)(i, j) << " ";
            }
            std::cout << "]\n";
        }
        std::cout << "\n";
    }

    void guardarArchivo(std::ofstream& out) const {
        out << filas_ << " " << columnas_ << "\n";
        for (size_t i = 0; i < filas_ * columnas_; i++) {
            out << datos_[i] << " ";
        }
        out << "\n";
    }

    void cargarArchivo(std::ifstream& in) {
        in >> filas_ >> columnas_;
        datos_.resize(filas_ * columnas_);
        for (size_t i = 0; i < filas_ * columnas_; ++i) {
            in >> datos_[i];
        }
    }

    void guardarBinario(std::ofstream& out) const {
        out.write(reinterpret_cast<const char*>(&filas_), sizeof(filas_));
        out.write(reinterpret_cast<const char*>(&columnas_), sizeof(columnas_));
        out.write(reinterpret_cast<const char*>(datos_.data()), datos_.size() * sizeof(float));
    }

    void cargarBinario(std::ifstream& in) {
        in.read(reinterpret_cast<char*>(&filas_), sizeof(filas_));
        in.read(reinterpret_cast<char*>(&columnas_), sizeof(columnas_));
        datos_.resize(filas_ * columnas_);
        in.read(reinterpret_cast<char*>(datos_.data()), datos_.size() * sizeof(float));
    }
};

#endif // MATRIX_HPP
