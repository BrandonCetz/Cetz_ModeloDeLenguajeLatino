#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include "RedNeuronal.hpp"
#include "matrix.hpp"
#include "capaDensa.hpp"

using namespace std;

int main() {
    Matriz X0(2,1); X0(0,0) = 0; X0(1,0) = 0;
    Matriz X1(2,1); X1(0,0) = 1; X1(1,0) = 0;
    Matriz X2(2,1); X2(0,0) = 0; X2(1,0) = 1;
    Matriz X3(2,1); X3(0,0) = 1; X3(1,0) = 1;

    Matriz Y0(1,1); Y0(0,0) = 0;
    Matriz Y1(1,1); Y1(0,0) = 1;
    Matriz Y2(1,1); Y2(0,0) = 1;
    Matriz Y3(1,1); Y3(0,0) = 0;

    vector<Matriz> entradas = {X0,X1,X2,X3};
    vector<Matriz> objetivos = {Y0,Y1,Y2,Y3};

    RedNeuronal red1;

    // Capa oculta ancha: 2 entradas -> 64 neuronas
    red1.agregarCapa(capaDensa(2, 4, ReLU, derivadaReLU));

    // Capa de salida: 64 neuronas -> 1 salida
    red1.agregarCapa(capaDensa(4, 1, sigmoide, derivadaSigmoide));

    red1.entrenar(entradas,objetivos,4000,0.01);

    red1.guardarModelo("Modelo_XOR.txt");
    for (size_t i = 0; i < entradas.size(); ++i) {
        Matriz pred = red1.forward(entradas[i]);
        cout << "Entrada: (" << entradas[i](0,0) << ", " << entradas[i](1,0)
        << ") -> Prediccion: " << pred(0,0)
        << " | Esperado: " << objetivos[i](0,0) << "\n";
    }



    return 0;
}
