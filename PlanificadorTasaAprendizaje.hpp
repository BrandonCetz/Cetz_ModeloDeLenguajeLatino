#pragma once
#include <cmath>

class PlanificadorTasaAprendizaje {
private:
    float tasaMax_;
    float tasaMin_;
    int pasosCalentamiento_;
    int pasosTotales_;

public:
    // tasaMax: Tasa máxima de aprendizaje (ej. 0.0020)
    // tasaMin: Tasa mínima al final del entrenamiento (ej. 0.0001)
    // pasosCalentamiento: Cuántos pasos tomará llegar al máximo
    // pasosTotales: Total de pasos de entrenamiento (épocas * muestras)
    PlanificadorTasaAprendizaje(float tasaMax, float tasaMin, int pasosCalentamiento, int pasosTotales)
        : tasaMax_(tasaMax), tasaMin_(tasaMin), pasosCalentamiento_(pasosCalentamiento), pasosTotales_(pasosTotales) {}

    float obtenerTasa(int pasoActual) const {
        // 1. Fase de Calentamiento (Warmup lineal)
        if (pasoActual < pasosCalentamiento_) {
            return tasaMax_ * (static_cast<float>(pasoActual) / static_cast<float>(pasosCalentamiento_));
        }

        // 2. Si superamos el límite, mantenemos el mínimo
        if (pasoActual >= pasosTotales_) {
            return tasaMin_;
        }

        // 3. Fase de Decaimiento Cosenoidal (Cosine Decay)
        float proporcionDecaimiento = static_cast<float>(pasoActual - pasosCalentamiento_) /
                                       static_cast<float>(pasosTotales_ - pasosCalentamiento_);

        // El coeficiente va de 1.0 a 0.0 siguiendo la mitad de un coseno
        float coeficiente = 0.5 * (1.0 + std::cos(3.14159265358979323846 * proporcionDecaimiento));

        return tasaMin_ + coeficiente * (tasaMax_ - tasaMin_);
    }
};
