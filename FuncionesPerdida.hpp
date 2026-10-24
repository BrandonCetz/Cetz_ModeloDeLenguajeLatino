#ifndef FUNCIONES_PERDIDA_HPP
#define FUNCIONES_PERDIDA_HPP

#include "matrix.hpp"
#include <cmath>
#include <algorithm>

// Pérdida de Entropía Cruzada Categórica para un token objetivo: -log(P[target])
inline float calcularCrossEntropy(const Matriz& prediccion, size_t targetId) {
    float p = (targetId < prediccion.filas()) ? prediccion(targetId, 0) : 1e-12f;
    if (p < 1e-12f) p = 1e-12f;
    return -std::log(p);
}

// Gradiente del Error Cuadrático Medio (MSE) - Para Regresión
inline Matriz gradienteMSE(const Matriz& prediccion, const Matriz& objetivo) {
    return prediccion - objetivo;
}

// Gradiente de la Entropía Cruzada Binaria (BCE) - Para Clasificación Binaria
inline Matriz gradienteBCE(const Matriz& prediccion, const Matriz& objetivo) {
    Matriz gradiente(prediccion.filas(), prediccion.columnas());

    for (size_t f = 0; f < prediccion.filas(); ++f) {
        for (size_t c = 0; c < prediccion.columnas(); ++c) {
            float p = prediccion(f, c);
            float y = objetivo(f, c);

            // Protección contra log(0) o división por cero (recorte de seguridad)
            float epsilon = 1e-15f;
            if (p < epsilon) p = epsilon;
            if (p > 1.0f - epsilon) p = 1.0f - epsilon;

            gradiente(f, c) = (p - y) / (p * (1.0f - p));
        }
    }

    return gradiente;
}

// Gradiente de la Entropía Cruzada Categórica con Softmax: Retorna (prediccion - objetivo)
inline Matriz gradienteCrossEntropy(const Matriz& prediccion, const Matriz& objetivo) {
    return prediccion - objetivo;
}

#endif // FUNCIONES_PERDIDA_HPP
