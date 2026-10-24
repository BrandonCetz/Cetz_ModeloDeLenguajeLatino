#include <iostream>
#include "Matrix.hpp"
#include "capaDensa.hpp"

using namespace std;

int main() {
    // 1. Datos de entrada (X) y valor objetivo deseado (Y_real)
    Matriz X(2, 1);
    X(0, 0) = 0.5;
    X(1, 0) = -0.2; //[cite: 4]

    Matriz Y_real(1, 1);
    Y_real(0, 0) = 1.0; // Queremos que la red aprenda a responder 1.0 para este patron

    // 2. Instanciar las dos capas (Red 2 -> 3 -> 1)
    capaDensa capa1(2, 3);
    capaDensa capa2(3, 1); //

    // Hiperparámetros de entrenamiento
    double tasaAprendizaje = 0.5;
    int epocas = 2000;

    cout << "--- PREDICCION INICIAL (SIN ENTRENAR) ---" << endl;
    Matriz inicio = capa2.forward(capa1.forward(X)); //[cite: 5]
    inicio.imprimir();
    cout << "-----------------------------------------\n" << endl;

    // 3. Bucle principal de entrenamiento (Epocas)
    for (int i = 0; i < epocas; ++i) {
        // --- PASO A: Forward Pass ---
        Matriz A1 = capa1.forward(X);   // Dimensión: 3x1[cite: 5]
        Matriz A2 = capa2.forward(A1);  // Dimensión: 1x1[cite: 5]

        // --- PASO B: Calcular el Gradiente Inicial del Error (A2 - Y_real) ---
        Matriz gradienteInicial = A2 - Y_real;

        // --- PASO C: Backward Pass en Reversa ---
        // La capa 2 calcula su ajuste y nos devuelve dX2 (dimensión 3x1)
        Matriz dX2 = capa2.backward(gradienteInicial, tasaAprendizaje); //

        // La capa 1 recibe dX2 como su gradiente de entrada y ajusta sus pesos
        Matriz dX1 = capa1.backward(dX2, tasaAprendizaje); //

        // Imprimir el avance cada 200 épocas
        if (i % 200 == 0 || i == epocas - 1) {
            double errorMSE = 0.5 * (A2(0, 0) - Y_real(0, 0)) * (A2(0, 0) - Y_real(0, 0));
            cout << "Epoca " << i << " | Salida: " << A2(0, 0) << " | Error MSE: " << errorMSE << endl;
        }
    }

    cout << "\n--- PREDICCION FINAL (DESPUES DE ENTRENAR) ---" << endl;
    Matriz final = capa2.forward(capa1.forward(X));
    final.imprimir();
    cout << "----------------------------------------------" << endl;

    return 0;
}
