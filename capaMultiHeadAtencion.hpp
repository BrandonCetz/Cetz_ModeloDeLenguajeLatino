#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "matrix.hpp"

class capaMultiHeadAtencion {
private:
    size_t dimModel_;
    size_t numHeads_;
    size_t dimK_;

    Matriz Wq, Wk, Wv, Wo;
    Matriz mWq, vWq, mWk, vWk, mWv, vWv, mWo, vWo;

    // Acumuladores de gradientes para Mini-Batching
    Matriz acum_dWq, acum_dWk, acum_dWv, acum_dWo;

    Matriz X_cache;
    Matriz Q_cache, K_cache, V_cache;
    Matriz H_concat_cache;
    std::vector<Matriz> A_cache_heads;
    std::vector<Matriz> scores_cache_heads;

    // ---- KV-Cache para inferencia incremental ----
    // Por cada cabeza: columnas K y V acumuladas de tokens ya procesados
    std::vector<Matriz> kv_K_cache;   // [numHeads]  (dimK × N_acum)
    std::vector<Matriz> kv_V_cache;   // [numHeads]  (dimK × N_acum)
    size_t kv_posicion_ = 0;          // Tokens ya almacenados en el cache

    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;

public:
    capaMultiHeadAtencion(size_t dimModel, size_t numHeads)
        : dimModel_(dimModel), numHeads_(numHeads), dimK_(dimModel / numHeads)
    {
        Wq = Matriz(dimModel_, dimModel_);
        Wk = Matriz(dimModel_, dimModel_);
        Wv = Matriz(dimModel_, dimModel_);
        Wo = Matriz(dimModel_, dimModel_);

        mWq = Matriz(dimModel_, dimModel_, 0.0f); vWq = Matriz(dimModel_, dimModel_, 0.0f);
        mWk = Matriz(dimModel_, dimModel_, 0.0f); vWk = Matriz(dimModel_, dimModel_, 0.0f);
        mWv = Matriz(dimModel_, dimModel_, 0.0f); vWv = Matriz(dimModel_, dimModel_, 0.0f);
        mWo = Matriz(dimModel_, dimModel_, 0.0f); vWo = Matriz(dimModel_, dimModel_, 0.0f);

        // Inicializar acumuladores en ceros
        acum_dWq = Matriz(dimModel_, dimModel_, 0.0f);
        acum_dWk = Matriz(dimModel_, dimModel_, 0.0f);
        acum_dWv = Matriz(dimModel_, dimModel_, 0.0f);
        acum_dWo = Matriz(dimModel_, dimModel_, 0.0f);

        float limite = std::sqrt(6.0f / (dimModel_ + dimModel_));
        Wq.aleatorizar(-limite, limite);
        Wk.aleatorizar(-limite, limite);
        Wv.aleatorizar(-limite, limite);
        Wo.aleatorizar(-limite, limite);
    }

    Matriz forward(const Matriz& X) {
        X_cache = X;
        size_t seqLen = X.columnas();

        Q_cache = Wq.matmul(X);
        K_cache = Wk.matmul(X);
        V_cache = Wv.matmul(X);

        A_cache_heads.resize(numHeads_);
        scores_cache_heads.resize(numHeads_);

        Matriz H_concat(dimModel_, seqLen);
        float escala = std::sqrt(static_cast<float>(dimK_));

        for (size_t h = 0; h < numHeads_; ++h) {
            size_t offsetFila = h * dimK_;

            Matriz Qh(dimK_, seqLen), Kh(dimK_, seqLen), Vh(dimK_, seqLen);
            for (size_t i = 0; i < dimK_; ++i) {
                for (size_t j = 0; j < seqLen; ++j) {
                    Qh(i, j) = Q_cache(offsetFila + i, j);
                    Kh(i, j) = K_cache(offsetFila + i, j);
                    Vh(i, j) = V_cache(offsetFila + i, j);
                }
            }

            Matriz scores = Qh.transpuesta().matmul(Kh);
            for (size_t i = 0; i < seqLen; ++i) {
                for (size_t j = 0; j < seqLen; ++j) {
                    scores(i, j) /= escala;
                    if (j > i) scores(i, j) = -1e9f;
                }
            }
            scores_cache_heads[h] = scores;

            Matriz Ah(seqLen, seqLen);
            for (size_t i = 0; i < seqLen; ++i) {
                float maxVal = scores(i, 0);
                for (size_t j = 1; j < seqLen; ++j) {
                    if (scores(i, j) > maxVal) maxVal = scores(i, j);
                }

                float sumExp = 0.0f;
                for (size_t j = 0; j < seqLen; ++j) {
                    float expVal = std::exp(scores(i, j) - maxVal);
                    Ah(i, j) = expVal;
                    sumExp += expVal;
                }

                for (size_t j = 0; j < seqLen; ++j) {
                    Ah(i, j) /= sumExp;
                }
            }
            A_cache_heads[h] = Ah;

            Matriz headOut = Vh.matmul(Ah.transpuesta());

            for (size_t i = 0; i < dimK_; ++i) {
                for (size_t j = 0; j < seqLen; ++j) {
                    H_concat(offsetFila + i, j) = headOut(i, j);
                }
            }
        }

        H_concat_cache = H_concat;
        return Wo.matmul(H_concat);
    }

    // Acumula gradientes para Mini-Batching
    Matriz backward(const Matriz& gradSalida) {
        size_t seqLen = gradSalida.columnas();

        Matriz dWo = gradSalida.matmul(H_concat_cache.transpuesta());
        Matriz dH_concat = Wo.transpuesta().matmul(gradSalida);

        // 1. Acumular gradientes de Wo
        acum_dWo.sumarInPlace(dWo);

        Matriz dQ_total(dimModel_, seqLen, 0.0f);
        Matriz dK_total(dimModel_, seqLen, 0.0f);
        Matriz dV_total(dimModel_, seqLen, 0.0f);

        float escala = std::sqrt(static_cast<float>(dimK_));

        for (size_t h = 0; h < numHeads_; ++h) {
            size_t offsetFila = h * dimK_;

            Matriz dHead(dimK_, seqLen);
            Matriz Qh(dimK_, seqLen), Kh(dimK_, seqLen), Vh(dimK_, seqLen);
            for (size_t i = 0; i < dimK_; ++i) {
                for (size_t j = 0; j < seqLen; ++j) {
                    dHead(i, j) = dH_concat(offsetFila + i, j);
                    Qh(i, j) = Q_cache(offsetFila + i, j);
                    Kh(i, j) = K_cache(offsetFila + i, j);
                    Vh(i, j) = V_cache(offsetFila + i, j);
                }
            }

            Matriz Ah = A_cache_heads[h];
            Matriz dVh = dHead.matmul(Ah);
            Matriz dAh = dHead.transpuesta().matmul(Vh);

            Matriz dScores(seqLen, seqLen, 0.0f);
            for (size_t i = 0; i < seqLen; ++i) {
                float sum_dAh_Ah = 0.0f;
                for (size_t k = 0; k < seqLen; ++k) {
                    sum_dAh_Ah += dAh(i, k) * Ah(i, k);
                }

                for (size_t j = 0; j < seqLen; ++j) {
                    dScores(i, j) = (Ah(i, j) * (dAh(i, j) - sum_dAh_Ah)) / escala;
                }
            }

            Matriz dQh = Kh.matmul(dScores.transpuesta());
            Matriz dKh = Qh.matmul(dScores);

            for (size_t i = 0; i < dimK_; ++i) {
                for (size_t j = 0; j < seqLen; ++j) {
                    dQ_total(offsetFila + i, j) = dQh(i, j);
                    dK_total(offsetFila + i, j) = dKh(i, j);
                    dV_total(offsetFila + i, j) = dVh(i, j);
                }
            }
        }

        Matriz dWq = dQ_total.matmul(X_cache.transpuesta());
        Matriz dWk = dK_total.matmul(X_cache.transpuesta());
        Matriz dWv = dV_total.matmul(X_cache.transpuesta());

        // 2. Acumular gradientes de Wq, Wk, Wv
        acum_dWq.sumarInPlace(dWq);
        acum_dWk.sumarInPlace(dWk);
        acum_dWv.sumarInPlace(dWv);

        Matriz gradFinal = Wq.transpuesta().matmul(dQ_total);
        gradFinal.sumarInPlace(Wk.transpuesta().matmul(dK_total));
        gradFinal.sumarInPlace(Wv.transpuesta().matmul(dV_total));

        return gradFinal;
    }

    // Calcula suma de cuadrados de gradientes acumulados promediados (para Gradient Clipping)
    float calcularNormaGradientesCuadrada(float batchSize) const {
        auto sumaMatriz = [batchSize](const Matriz& M) {
            float suma = 0.0f;
            for (size_t i = 0; i < M.filas(); ++i) {
                for (size_t j = 0; j < M.columnas(); ++j) {
                    float g = M(i, j) / batchSize;
                    suma += g * g;
                }
            }
            return suma;
        };
        return sumaMatriz(acum_dWq) + sumaMatriz(acum_dWk) +
               sumaMatriz(acum_dWv) + sumaMatriz(acum_dWo);
    }

    // Escala gradientes acumulados por un factor (para Gradient Clipping)
    void escalarGradientesAcumulados(float factor) {
        auto escalar = [factor](Matriz& M) {
            for (size_t i = 0; i < M.filas(); ++i) {
                for (size_t j = 0; j < M.columnas(); ++j) {
                    M(i, j) *= factor;
                }
            }
        };
        escalar(acum_dWq);
        escalar(acum_dWk);
        escalar(acum_dWv);
        escalar(acum_dWo);
    }

    // Ejecuta Adam para las 4 matrices promediadas
    void aplicarOptimizador(float tasaDeAprendizaje, int t, float batchSize) {
        float corr1 = 1.0f - std::pow(beta1, t);
        float corr2 = 1.0f - std::pow(beta2, t);

        actualizarParametroAcumulado(Wq, acum_dWq, mWq, vWq, tasaDeAprendizaje, corr1, corr2, batchSize);
        actualizarParametroAcumulado(Wk, acum_dWk, mWk, vWk, tasaDeAprendizaje, corr1, corr2, batchSize);
        actualizarParametroAcumulado(Wv, acum_dWv, mWv, vWv, tasaDeAprendizaje, corr1, corr2, batchSize);
        actualizarParametroAcumulado(Wo, acum_dWo, mWo, vWo, tasaDeAprendizaje, corr1, corr2, batchSize);
    }

    void guardarBinario(std::ofstream& out) const {
        Wq.guardarBinario(out); Wk.guardarBinario(out);
        Wv.guardarBinario(out); Wo.guardarBinario(out);
    }

    void cargarBinario(std::ifstream& in) {
        Wq.cargarBinario(in); Wk.cargarBinario(in);
        Wv.cargarBinario(in); Wo.cargarBinario(in);
    }

    // ---- API de KV-Cache (solo inferencia) ----

    // Limpia el cache entre prompts distintos
    void limpiarKVCache() {
        kv_K_cache.clear();
        kv_V_cache.clear();
        kv_posicion_ = 0;
    }

    // Copia los K/V calculados en el ultimo forward() completo hacia el KV-Cache.
    // Debe llamarse justo despues de forward(contextoCompleto) para "sembrar" el cache.
    void llenarKVCacheDesdeForward(size_t seqLen) {
        kv_K_cache.resize(numHeads_);
        kv_V_cache.resize(numHeads_);

        for (size_t h = 0; h < numHeads_; ++h) {
            size_t offsetFila = h * dimK_;
            // Extraemos las columnas [0..seqLen) de K_cache y V_cache para la cabeza h
            Matriz Kh(dimK_, seqLen);
            Matriz Vh(dimK_, seqLen);
            for (size_t i = 0; i < dimK_; ++i) {
                for (size_t j = 0; j < seqLen; ++j) {
                    Kh(i, j) = K_cache(offsetFila + i, j);
                    Vh(i, j) = V_cache(offsetFila + i, j);
                }
            }
            kv_K_cache[h] = std::move(Kh);
            kv_V_cache[h] = std::move(Vh);
        }
        kv_posicion_ = seqLen;
    }

    // Procesa UN solo token nuevo (x_nuevo: dimModel×1) usando el KV-Cache acumulado.
    // Devuelve la salida de atencion para ese token (dimModel×1).
    Matriz forwardConCache(const Matriz& x_nuevo) {
        // Proyectar el token nuevo a Q, K, V globales
        Matriz q_nuevo = Wq.matmul(x_nuevo);   // (dimModel × 1)
        Matriz k_nuevo = Wk.matmul(x_nuevo);   // (dimModel × 1)
        Matriz v_nuevo = Wv.matmul(x_nuevo);   // (dimModel × 1)

        float escala = std::sqrt(static_cast<float>(dimK_));

        Matriz H_concat(dimModel_, 1, 0.0f);

        for (size_t h = 0; h < numHeads_; ++h) {
            size_t offsetFila = h * dimK_;

            // Extraer slice de la cabeza h para el token nuevo (dimK×1)
            Matriz qh(dimK_, 1);
            Matriz kh_nuevo(dimK_, 1);
            Matriz vh_nuevo(dimK_, 1);
            for (size_t i = 0; i < dimK_; ++i) {
                qh(i, 0)      = q_nuevo(offsetFila + i, 0);
                kh_nuevo(i, 0) = k_nuevo(offsetFila + i, 0);
                vh_nuevo(i, 0) = v_nuevo(offsetFila + i, 0);
            }

            // Agregar el nuevo K y V al cache de esta cabeza
            size_t N_viejo = kv_posicion_;  // tokens ya en cache
            size_t N_total = N_viejo + 1;

            // Construir K_full y V_full: (dimK × N_total)
            Matriz K_full(dimK_, N_total);
            Matriz V_full(dimK_, N_total);

            // Copiar columnas existentes del cache
            for (size_t i = 0; i < dimK_; ++i) {
                for (size_t j = 0; j < N_viejo; ++j) {
                    K_full(i, j) = kv_K_cache[h](i, j);
                    V_full(i, j) = kv_V_cache[h](i, j);
                }
                // Agregar nueva columna
                K_full(i, N_viejo) = kh_nuevo(i, 0);
                V_full(i, N_viejo) = vh_nuevo(i, 0);
            }

            // Actualizar el cache con la nueva columna (extender la matriz existente)
            kv_K_cache[h] = K_full;
            kv_V_cache[h] = V_full;

            // Attention: qh^T * K_full → (1 × N_total)
            // scores[j] = sum_i qh[i] * K_full[i,j]
            std::vector<float> scores(N_total, 0.0f);
            for (size_t j = 0; j < N_total; ++j) {
                float s = 0.0f;
                for (size_t i = 0; i < dimK_; ++i) {
                    s += qh(i, 0) * K_full(i, j);
                }
                scores[j] = s / escala;
            }

            // Softmax numericamente estable sobre (N_total) scores
            float maxS = *std::max_element(scores.begin(), scores.end());
            float sumExp = 0.0f;
            for (size_t j = 0; j < N_total; ++j) {
                scores[j] = std::exp(scores[j] - maxS);
                sumExp += scores[j];
            }
            for (size_t j = 0; j < N_total; ++j) scores[j] /= sumExp;

            // Vector de contexto: V_full * attention_weights → (dimK × 1)
            // ctx[i] = sum_j V_full[i,j] * scores[j]
            std::vector<float> ctx(dimK_, 0.0f);
            for (size_t i = 0; i < dimK_; ++i) {
                for (size_t j = 0; j < N_total; ++j) {
                    ctx[i] += V_full(i, j) * scores[j];
                }
            }

            // Escribir en la parte correspondiente de H_concat
            for (size_t i = 0; i < dimK_; ++i) {
                H_concat(offsetFila + i, 0) = ctx[i];
            }
        }

        // Actualizar la posicion SOLO UNA VEZ despues de procesar todas las cabezas
        kv_posicion_ += 1;

        return Wo.matmul(H_concat);   // (dimModel × 1)
    }

private:
    void actualizarParametroAcumulado(Matriz& W, Matriz& acum_dW, Matriz& mW, Matriz& vW, float lr, float corr1, float corr2, float batchSize) {
        size_t filas = W.filas();
        size_t cols = W.columnas();

        for (size_t i = 0; i < filas; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                float g = acum_dW(i, j) / batchSize;

                float m_val = beta1 * mW(i, j) + (1.0f - beta1) * g;
                float v_val = beta2 * vW(i, j) + (1.0f - beta2) * (g * g);

                mW(i, j) = m_val;
                vW(i, j) = v_val;

                float m_hat = m_val / corr1;
                float v_hat = v_val / corr2;

                W(i, j) -= (lr * m_hat) / (std::sqrt(v_hat) + epsilon);

                acum_dW(i, j) = 0.0f;
            }
        }
    }
};
