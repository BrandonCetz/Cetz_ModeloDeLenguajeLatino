#pragma once
#include <iostream>
#include "matrix.hpp"
#include "capaMultiHeadAtencion.hpp"
#include "capaDensa.hpp"
#include "LayerNorm.hpp"

class BloqueTransformer {
private:
    capaMultiHeadAtencion atencion;
    LayerNorm norm1;
    LayerNorm norm2;

    capaDensa ffn1;
    capaDensa ffn2;

    Matriz X_cache;
    Matriz xNorm1_cache;
    Matriz res1_cache;
    Matriz xNorm2_cache;

public:
    BloqueTransformer(size_t dimModel, size_t numHeads, size_t dimFFN)
        : atencion(dimModel, numHeads),
          norm1(dimModel), norm2(dimModel),
          ffn1(dimModel, dimFFN, ReLU, derivadaReLU),
          ffn2(dimFFN, dimModel, lineal, derivadaLineal)
    {}

    // Forward Pass con arquitectura Pre-LayerNorm moderna
    Matriz forward(const Matriz& X) {
        X_cache = X;

        // 1. Subcapa 1: Pre-LN + Atención Multicabezal + Conexión Residual
        xNorm1_cache = norm1.forward(X);
        Matriz atnOut = atencion.forward(xNorm1_cache);
        res1_cache = X + atnOut;

        // 2. Subcapa 2: Pre-LN + Feed-Forward (ReLU -> Lineal) + Conexión Residual
        xNorm2_cache = norm2.forward(res1_cache);
        Matriz h = ffn1.forward(xNorm2_cache);
        Matriz ffnOut = ffn2.forward(h);
        Matriz res2 = res1_cache + ffnOut;

        return res2;
    }

    // Backward Pass con retropropagación exacta a través de Normalizaciones y Residuales
    Matriz backward(const Matriz& gradSalida) {
        // 1. Gradiente a través de la FFN
        Matriz dh2 = ffn2.backward(gradSalida);
        Matriz dNorm2 = ffn1.backward(dh2);

        // 2. Gradiente a través de norm2
        Matriz dRes1_via_ffn = norm2.backward(dNorm2);

        // 3. Gradiente total en el punto res1 (autopista residual directa + camino FFN)
        Matriz dRes1 = gradSalida + dRes1_via_ffn;

        // 4. Gradiente a través de la Atención
        Matriz dAtn = atencion.backward(dRes1);

        // 5. Gradiente a través de norm1
        Matriz dX_via_atn = norm1.backward(dAtn);

        // 6. Gradiente total en la entrada X (autopista residual directa + camino Atención)
        return dRes1 + dX_via_atn;
    }

    // Suma de cuadrados de gradientes de todas sus subcapas (para Gradient Clipping)
    float calcularNormaGradientesCuadrada(float batchSize) const {
        return atencion.calcularNormaGradientesCuadrada(batchSize) +
               ffn1.calcularNormaGradientesCuadrada(batchSize) +
               ffn2.calcularNormaGradientesCuadrada(batchSize);
    }

    // Escala gradientes de todas sus subcapas (para Gradient Clipping)
    void escalarGradientesAcumulados(float factor) {
        atencion.escalarGradientesAcumulados(factor);
        ffn1.escalarGradientesAcumulados(factor);
        ffn2.escalarGradientesAcumulados(factor);
    }

    // Pasa la señal de actualización a sus subcapas
    void aplicarOptimizador(float lr, int t, float batchSize) {
        atencion.aplicarOptimizador(lr, t, batchSize);
        ffn1.aplicarOptimizador(lr, t, batchSize);
        ffn2.aplicarOptimizador(lr, t, batchSize);
    }

    void guardarBinario(std::ofstream& out) const {
        atencion.guardarBinario(out); ffn1.guardarBinario(out); ffn2.guardarBinario(out);
    }
    void cargarBinario(std::ifstream& in) {
        atencion.cargarBinario(in); ffn1.cargarBinario(in); ffn2.cargarBinario(in);
    }

    // ---- API de KV-Cache (solo inferencia) ----

    // Limpia el cache de atencion entre sesiones de inferencia
    void limpiarKVCache() {
        atencion.limpiarKVCache();
    }

    // Transfiere los K/V del ultimo forward() completo al KV-Cache de atencion
    void llenarKVCacheDesdeForward(size_t seqLen) {
        atencion.llenarKVCacheDesdeForward(seqLen);
    }

    // Procesa UN token nuevo (x_nuevo: dimModel×1) de forma incremental con Pre-LN
    Matriz forwardConCache(const Matriz& x_nuevo) {
        // Pre-LN + Atencion con KV-Cache + Residual
        Matriz xNorm1 = norm1.forward(x_nuevo);
        Matriz atnOut = atencion.forwardConCache(xNorm1);
        Matriz res1   = x_nuevo + atnOut;

        // Pre-LN + FFN + Residual
        Matriz xNorm2  = norm2.forward(res1);
        Matriz h       = ffn1.forward(xNorm2);
        Matriz ffnOut  = ffn2.forward(h);
        return res1 + ffnOut;
    }
};
