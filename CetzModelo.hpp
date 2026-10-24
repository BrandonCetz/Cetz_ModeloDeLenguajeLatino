#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <random>

#include "matrix.hpp"
#include "capaEmbedding.hpp"
#include "PositionalEncoding.hpp"
#include "BloqueTransformer.hpp"
#include "LayerNorm.hpp"
#include "capaDensa.hpp"
#include "FuncionesPerdida.hpp"

class CetzModelo {
private:
    size_t vocabSize_;
    size_t dimModel_;
    size_t maxSeqLen_;
    size_t numBloques_;

    capaEmbedding emb_;
    PositionalEncoding posEnc_;
    std::vector<BloqueTransformer> bloques_;
    LayerNorm normFinal_;
    capaDensa lmHead_;

    // ---- Estado de la sesion de inferencia con KV-Cache ----
    std::vector<size_t> contexto_cache_;  // tokens del contexto activo (para penalizacion de repeticion)
    size_t tokens_en_cache_ = 0;          // cuantos tokens estan ya en el KV-Cache

    // Motor de Muestreo Estocástico / Determinista Optimizado
    size_t muestrearToken(const Matriz& probas, float temperatura, int topK, float topP) {
        size_t n = probas.filas();

        // Modo Greedy Estricto
        if (temperatura <= 0.05f || topK == 1) {
            size_t idMejor = 0;
            float maxP = probas(0, 0);
            for (size_t i = 1; i < n; ++i) {
                if (probas(i, 0) > maxP) {
                    maxP = probas(i, 0);
                    idMejor = i;
                }
            }
            return idMejor;
        }

        std::vector<std::pair<float, size_t>> items;
        float suma = 0.0f;

        for (size_t i = 0; i < n; ++i) {
            float p = std::pow(probas(i, 0), 1.0f / temperatura);
            items.push_back({p, i});
            suma += p;
        }

        for (auto& item : items) item.first /= suma;
        std::sort(items.rbegin(), items.rend());

        // Aplicar Top-K
        if (topK > 0 && static_cast<size_t>(topK) < items.size()) {
            items.resize(topK);
        }

        // Aplicar Top-P (Nucleus Sampling real)
        if (topP < 1.0f) {
            float acum = 0.0f;
            size_t corte = items.size();
            for (size_t i = 0; i < items.size(); ++i) {
                acum += items[i].first;
                if (acum >= topP) {
                    corte = i + 1;
                    break;
                }
            }
            items.resize(corte);
        }

        // Renormalizar probabilidades finales
        float sumaFiltrada = 0.0f;
        for (const auto& item : items) sumaFiltrada += item.first;

        std::vector<float> probasFinales;
        std::vector<size_t> indicesFinales;
        for (const auto& item : items) {
            probasFinales.push_back(item.first / sumaFiltrada);
            indicesFinales.push_back(item.second);
        }

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::discrete_distribution<size_t> dist(probasFinales.begin(), probasFinales.end());

        return indicesFinales[dist(gen)];
    }

public:
    CetzModelo(size_t vocabSize, size_t dimModel, size_t numHeads, size_t dimFFN, size_t numBloques, size_t maxSeqLen)
        : vocabSize_(vocabSize),
          dimModel_(dimModel),
          maxSeqLen_(maxSeqLen),
          numBloques_(numBloques),
          emb_(vocabSize, dimModel),
          posEnc_(maxSeqLen, dimModel),
          normFinal_(dimModel),
          lmHead_(dimModel, vocabSize, sigmoide, derivadaSigmoide)
    {
        lmHead_.usarSoftmax();
        for (size_t b = 0; b < numBloques_; ++b) {
            bloques_.emplace_back(dimModel, numHeads, dimFFN);
        }
    }

    ~CetzModelo() {}

    // Forward Pass con arquitectura Pre-LN y ventana de contexto deslizante
    Matriz forward(const std::vector<size_t>& seqEntrada) {
        std::vector<size_t> seq = seqEntrada;
        if (seq.size() > maxSeqLen_) {
            // Ventana deslizante: Conservar los últimos maxSeqLen_ tokens
            seq.erase(seq.begin(), seq.begin() + (seq.size() - maxSeqLen_));
        }

        size_t seqLen = seq.size();
        Matriz X(dimModel_, seqLen);

        for (size_t pos = 0; pos < seqLen; ++pos) {
            Matriz vEmb = emb_.forward(seq[pos]);
            Matriz vPos = posEnc_.obtenerPosicion(pos);
            vEmb.sumarInPlace(vPos); // Optimizado: In-Place
            for (size_t d = 0; d < dimModel_; ++d) {
                X(d, pos) = vEmb(d, 0);
            }
        }

        Matriz act = X;
        for (size_t b = 0; b < numBloques_; ++b) {
            act = bloques_[b].forward(act);
        }

        // Normalización final Pre-LN antes de proyectar hacia el vocabulario
        act = normFinal_.forward(act);

        return act;
    }

    // Calcula y acumula el gradiente de la secuencia, retornando el Loss promedio
    float trainStepAcumulativo(const std::vector<size_t>& seqEntrada, const std::vector<size_t>& seqObjetivo) {
        size_t seqLen = seqEntrada.size();
        if (seqLen == 0) return 0.0f;

        // 1. Forward Pass Transformer
        Matriz act = forward(seqEntrada);

        // 2. LM Head Forward vectorizado en una sola operación para toda la secuencia
        Matriz pred = lmHead_.forward(act);

        // 3. Pérdida y Gradiente Analítico Directo
        float lossTotal = 0.0f;
        Matriz gradCross = pred;

        for (size_t pos = 0; pos < seqLen; ++pos) {
            size_t targetId = seqObjetivo[pos];
            if (targetId < vocabSize_) {
                float prob = pred(targetId, pos);
                if (prob < 1e-12f) prob = 1e-12f;
                lossTotal += -std::log(prob);
                gradCross(targetId, pos) -= 1.0f;
            }
        }

        // 4. LM Head Backward vectorizado en una sola operación
        Matriz gradActual = lmHead_.backward(gradCross);

        // Backward Pass a través de la Normalización Final Pre-LN
        gradActual = normFinal_.backward(gradActual);

        // Backward Pass a través de los Bloques Transformer
        for (int b = static_cast<int>(numBloques_) - 1; b >= 0; --b) {
            gradActual = bloques_[b].backward(gradActual);
        }

        // Backward Pass a través de Embeddings
        for (size_t pos = 0; pos < seqLen; ++pos) {
            Matriz gradEmb(dimModel_, 1);
            for (size_t d = 0; d < dimModel_; ++d) {
                gradEmb(d, 0) = gradActual(d, pos);
            }
            emb_.backward(gradEmb, seqEntrada[pos]);
        }

        return lossTotal / static_cast<float>(seqLen);
    }

    // Recorta la norma global de los gradientes para estabilizar el entrenamiento (Gradient Clipping)
    void recortarGradientes(float maxNorma, float batchSize) {
        float sumaCuadrados = 0.0f;
        sumaCuadrados += lmHead_.calcularNormaGradientesCuadrada(batchSize);
        for (const auto& bloque : bloques_) {
            sumaCuadrados += bloque.calcularNormaGradientesCuadrada(batchSize);
        }
        sumaCuadrados += emb_.calcularNormaGradientesCuadrada(batchSize);

        float normaGlobal = std::sqrt(sumaCuadrados);
        if (normaGlobal > maxNorma && normaGlobal > 1e-6f) {
            float factor = maxNorma / normaGlobal;
            lmHead_.escalarGradientesAcumulados(factor);
            for (auto& bloque : bloques_) {
                bloque.escalarGradientesAcumulados(factor);
            }
            emb_.escalarGradientesAcumulados(factor);
        }
    }

    // Aplica Adam con Gradient Clipping (maxNorma = 1.0 por defecto)
    void actualizarPesos(float lr, int t, float batchSize, float maxNorma = 1.0f) {
        if (maxNorma > 0.0f) {
            recortarGradientes(maxNorma, batchSize);
        }
        lmHead_.aplicarOptimizador(lr, t, batchSize);
        for (size_t b = 0; b < numBloques_; ++b) {
            bloques_[b].aplicarOptimizador(lr, t, batchSize);
        }
        emb_.aplicarOptimizador(lr, t, batchSize);
    }

    // Inferencia con Penalización de Repetición (evita bucles infinitos)
    size_t predecirSiguiente(const std::vector<size_t>& contexto, float temperatura, int topK, float topP, float repetitionPenalty = 1.3f) {
        size_t seqLen = contexto.size();
        Matriz act = forward(contexto);

        // Si se aplicó ventana deslizante en forward, el último vector está en act.columnas() - 1
        size_t colUltimo = act.columnas() - 1;
        Matriz ultimoVec(dimModel_, 1);
        for (size_t d = 0; d < dimModel_; ++d) {
            ultimoVec(d, 0) = act(d, colUltimo);
        }

        Matriz pred = lmHead_.forward(ultimoVec);

        // Aplicar penalización a los últimos 10 tokens del contexto
        size_t ventanaHistorial = (std::min)(seqLen, static_cast<size_t>(10));
        for (size_t i = seqLen - ventanaHistorial; i < seqLen; ++i) {
            size_t idRepetido = contexto[i];
            if (idRepetido < pred.filas()) {
                pred(idRepetido, 0) /= repetitionPenalty;
            }
        }

        return muestrearToken(pred, temperatura, topK, topP);
    }

    // ============================================================
    // API de KV-Cache — solo para inferencia (no afecta training)
    // ============================================================

    // Inicializa el KV-Cache procesando el contexto completo del prompt.
    // Hace un forward() completo para calcular K/V de todos los tokens del prompt
    // y los "siembra" en el cache de cada bloque. Los tokens siguientes se
    // generan incrementalmente, uno por uno, sin recalcular los anteriores.
    void iniciarSesionInferencia(const std::vector<size_t>& contexto) {
        // Limpiar cache anterior si existe
        limpiarCache();

        // Guardar contexto para la penalizacion de repeticion
        contexto_cache_ = contexto;

        // Aplicar ventana deslizante si el contexto excede maxSeqLen_
        std::vector<size_t> seq = contexto;
        if (seq.size() > maxSeqLen_) {
            seq.erase(seq.begin(), seq.begin() + (seq.size() - maxSeqLen_));
        }
        size_t seqLen = seq.size();

        // 1. Forward completo del Transformer (esto puebla K_cache y V_cache internos)
        forward(seq);

        // 2. Trasladar los K/V calculados al KV-Cache de cada bloque
        for (size_t b = 0; b < numBloques_; ++b) {
            bloques_[b].llenarKVCacheDesdeForward(seqLen);
        }

        tokens_en_cache_ = seqLen;
    }

    // Predice el siguiente token usando el KV-Cache (proceso incremental O(1) por token).
    // Solo calcula Q, K, V para el ULTIMO token; los K/V de los anteriores ya estan cacheados.
    size_t predecirSiguienteConCache(float temperatura, int topK, float topP, float repetitionPenalty = 1.3f) {
        if (tokens_en_cache_ == 0) return 0;

        // El ultimo token del contexto actual es el que ingresamos como nuevo token
        size_t idUltimo = contexto_cache_.back();
        size_t posActual = tokens_en_cache_;

        // 1. Embedding + Encoding Posicional para el token nuevo (posicion = posActual)
        Matriz vEmb = emb_.forward(idUltimo);
        Matriz vPos = posEnc_.obtenerPosicion(posActual < maxSeqLen_ ? posActual : maxSeqLen_ - 1);
        Matriz x_nuevo = vEmb + vPos;   // (dimModel × 1)

        // 2. Pasar por cada bloque con KV-Cache incremental
        Matriz act = x_nuevo;
        for (size_t b = 0; b < numBloques_; ++b) {
            act = bloques_[b].forwardConCache(act);
        }

        // 3. Normalizacion final + LM Head
        act = normFinal_.forward(act);    // (dimModel × 1)
        Matriz pred = lmHead_.forward(act);  // (vocabSize × 1)

        // 4. Penalizacion de repeticion: ultimos 10 tokens del contexto
        size_t historialLen = contexto_cache_.size();
        size_t ventana = (std::min)(historialLen, static_cast<size_t>(10));
        for (size_t i = historialLen - ventana; i < historialLen; ++i) {
            size_t idRep = contexto_cache_[i];
            if (idRep < pred.filas()) {
                pred(idRep, 0) /= repetitionPenalty;
            }
        }

        // 5. Muestrear el siguiente token
        size_t idSiguiente = muestrearToken(pred, temperatura, topK, topP);

        // 6. Actualizar el contexto y el contador de cache para la proxima llamada
        contexto_cache_.push_back(idSiguiente);
        tokens_en_cache_++;

        return idSiguiente;
    }

    // Resetea el KV-Cache entre prompts
    void limpiarCache() {
        for (size_t b = 0; b < numBloques_; ++b) {
            bloques_[b].limpiarKVCache();
        }
        contexto_cache_.clear();
        tokens_en_cache_ = 0;
    }

    // Métodos de Persistencia
    bool guardar(const std::string& ruta) const {
        std::ofstream out(ruta, std::ios::binary);
        if (!out.is_open()) return false;
        emb_.guardarBinario(out);
        for (size_t b = 0; b < numBloques_; ++b) {
            bloques_[b].guardarBinario(out);
        }
        lmHead_.guardarBinario(out);
        out.close();
        return true;
    }

    bool cargar(const std::string& ruta) {
        std::ifstream in(ruta, std::ios::binary);
        if (!in.is_open()) return false;
        emb_.cargarBinario(in);
        for (size_t b = 0; b < numBloques_; ++b) {
            bloques_[b].cargarBinario(in);
        }
        lmHead_.cargarBinario(in);
        in.close();
        return true;
    }
};
