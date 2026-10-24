#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <iomanip>
#include <cmath>
#include <windows.h>
#include "PlanificadorTasaAprendizaje.hpp"
#include "TokenizadorSubPalabras.hpp"
#include "CetzModelo.hpp"

using namespace std;

vector<string> cargarCorpusMasivo(const string& ruta, size_t maxPalabras = 60) {
    vector<string> dataset;
    ifstream in(ruta);
    if (!in.is_open()) {
        cerr << "[ERROR] No se pudo abrir " << ruta << endl;
        return dataset;
    }

    string palabra;
    vector<string> bufferFrase;

    while (in >> palabra) {
        bufferFrase.push_back(palabra);

        char ultimoChar = palabra.back();
        if (ultimoChar == '.' || ultimoChar == '?' || ultimoChar == '!' || bufferFrase.size() >= maxPalabras) {
            string frase = "";
            for (const string& p : bufferFrase) frase += p + " ";
            dataset.push_back(frase);
            bufferFrase.clear();
        }
    }

    if (!bufferFrase.empty()) {
        string frase = "";
        for (const string& p : bufferFrase) frase += p + " ";
        dataset.push_back(frase);
        bufferFrase.clear();
    }

    in.close();

    cout << "[INFO] Total de oraciones/muestras generadas: " << dataset.size() << endl;
    return dataset;
}

int main() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    // Parámetros de Inferencia y Muestreo (Afinados para coherencia y estilo)
    float temperatura = 0.75f;
    int topK = 30;
    float topP = 0.85f;
    float penalizacion = 1.25f;

    // Hiperparámetros de Arquitectura Transformer (Cetz v2.0)
    size_t dimModel = 128;      // Capacidad semántica aumentada (128 dimensiones)
    size_t numHeads = 4;        // 4 cabezas de atención (dimK = 32 por cabeza)
    size_t dimFFN = 512;        // Feed-Forward expandido (2x dimModel)
    size_t numBloques = 2;      // 2 Bloques Pre-LN balanceados para entrenamiento veloz
    size_t maxSeqLen = 128;      // Ventana de contexto

    // Parámetros de Entrenamiento
    int epocas = 40;            // 35 épocas para alcanzar la zona de convergencia
    int t = 1;
    float batchSize = 16.0f;

    float tasaMax = 0.0005f;    // Tasa óptima con calentamiento y clipping
    float tasaMin = 0.00005f;

    int pasoActual = 0;
    int contadorBatch = 0;

    size_t limiteVocabulario = 2000; // Vocabulario ampliado (palabras completas en vez de sílabas)

    string archivoCorpus = "corpus_mex.txt";
    string archivoPesos = "cetz_words_Cmex.bin";
    string archivoVocab = "cetz_vocab_Cmex.txt";

    TokenizadorSubPalabras tokenizador;

    cout << "==================================================" << endl;
    cout << "   CETZ-LM (Transformer en Español Latino - v2.0)" << endl;
    cout << "   - Arquitectura: Pre-LN, " << numBloques << " Bloques, Dim: " << dimModel << endl;
    cout << "   - Optimización: OpenMP Multi-Hilo + FFN Lineal" << endl;
    cout << "==================================================" << endl;
    cout << "1. Entrenar y guardar nuevo modelo (35 épocas)" << endl;
    cout << "2. Cargar modelo y vocabulario existentes" << endl;
    cout << "Opción > ";

    int opcion = 1;
    cin >> opcion;
    cin.ignore();

    unique_ptr<CetzModelo> cetzModelo;

    if (opcion == 1) {
        vector<string> dataset = cargarCorpusMasivo(archivoCorpus, 50);

        if (dataset.empty()) {
            cout << "[AVISO]: '" << archivoCorpus << "' no encontrado. Usando corpus de prueba..." << endl;
            dataset = {
                "HOLA COMO ESTAS TU HOY",
                "EL MODELO CETZ APRENDE PALABRAS EN LUGAR DE LETRAS",
                "CETZ ES UN TRANSFORMER C++ ULTRA RAPIDO",
                "ME GUSTA PROGRAMAR INTELIGENCIA ARTIFICIAL EN C++",
                "LA TIERRA GIRA ALREDEDOR DEL SOL"
            };
        }

        cout << "[INFO] Construyendo vocabulario BPE (límite: " << limiteVocabulario << " tokens)..." << endl;
        tokenizador.construirVocabulario(dataset, limiteVocabulario);
        tokenizador.guardar(archivoVocab);
        cout << "[INFO] Vocabulario guardado en: " << archivoVocab << " (" << tokenizador.vocabSize() << " tokens únicos)" << endl;

        size_t vocabSize = tokenizador.vocabSize();
        cetzModelo = make_unique<CetzModelo>(vocabSize, dimModel, numHeads, dimFFN, numBloques, maxSeqLen);

        struct Muestra {
            vector<size_t> entrada;
            vector<size_t> objetivo;
        };
        vector<Muestra> datasetCodificado;

        for (const string& frase : dataset) {
            vector<size_t> seq = tokenizador.codificar(frase);
            if (seq.size() < 2) continue;

            if (seq.size() > maxSeqLen - 1) {
                seq.resize(maxSeqLen - 1);
            }

            vector<size_t> entrada = seq;
            vector<size_t> objetivo = seq;

            objetivo.erase(objetivo.begin());
            objetivo.push_back(tokenizador.idEos);

            datasetCodificado.push_back({entrada, objetivo});
        }

        int totalPasos = (epocas * datasetCodificado.size()) / static_cast<int>(batchSize);
        int pasosCalentamiento = static_cast<int>(totalPasos * 0.1);
        PlanificadorTasaAprendizaje planificador(tasaMax, tasaMin, pasosCalentamiento, totalPasos);

        cout << "\nEntrenando Cetz con " << datasetCodificado.size() << " muestras en Batches de " << batchSize << "..." << endl;
        cout << "Optimizador: Adam con Warmup + Gradient Clipping (maxNorm = 1.0)\n" << endl;

        for (int e = 0; e < epocas; ++e) {
            float tasaActual = 0.0f;
            float lossTotalEpoca = 0.0f;
            int totalMuestras = 0;

            for (const auto& muestra : datasetCodificado) {
                float pasoLoss = cetzModelo->trainStepAcumulativo(muestra.entrada, muestra.objetivo);
                lossTotalEpoca += pasoLoss;
                totalMuestras++;
                contadorBatch++;

                if (contadorBatch == static_cast<int>(batchSize)) {
                    tasaActual = planificador.obtenerTasa(pasoActual);
                    cetzModelo->actualizarPesos(tasaActual, t, batchSize, 1.0f); // Gradient Clipping
                    t++;
                    pasoActual++;
                    contadorBatch = 0;
                }
            }

            if (contadorBatch > 0) {
                tasaActual = planificador.obtenerTasa(pasoActual);
                cetzModelo->actualizarPesos(tasaActual, t, static_cast<float>(contadorBatch), 1.0f);
                t++;
                pasoActual++;
                contadorBatch = 0;
            }

            float lossPromedio = (totalMuestras > 0) ? (lossTotalEpoca / static_cast<float>(totalMuestras)) : 0.0f;
            float perplejidad = std::exp((std::min)(lossPromedio, 20.0f));

            if ((e + 1) % 2 == 0 || e == 0 || e == epocas - 1) {
                cout << "  Progreso: Época " << setw(2) << (e + 1) << " / " << epocas
                     << " | Loss Promedio: " << fixed << setprecision(4) << lossPromedio
                     << " | Perplejidad: " << setw(6) << setprecision(2) << perplejidad
                     << " | LR: " << scientific << setprecision(2) << tasaActual
                     << defaultfloat << endl;
            }
        }

        cetzModelo->guardar(archivoPesos);
        cout << "\n¡Modelo guardado exitosamente en " << archivoPesos << "!" << endl;

    } else {
        cout << "[INFO] Cargando vocabulario desde " << archivoVocab << "..." << endl;
        if (!tokenizador.cargar(archivoVocab)) {
            cerr << "[ERROR] No se pudo cargar el vocabulario '" << archivoVocab << "'. Asegúrate de entrenar primero." << endl;
            return 1;
        }
        cout << "[INFO] Vocabulario cargado con éxito (" << tokenizador.vocabSize() << " tokens únicos)." << endl;

        size_t vocabSize = tokenizador.vocabSize();
        cetzModelo = make_unique<CetzModelo>(vocabSize, dimModel, numHeads, dimFFN, numBloques, maxSeqLen);

        cout << "[INFO] Cargando pesos del modelo desde " << archivoPesos << "..." << endl;
        if (!cetzModelo->cargar(archivoPesos)) {
            cerr << "[ERROR] No se pudo cargar el archivo de pesos '" << archivoPesos << "'." << endl;
            return 1;
        }
        cout << "[INFO] Modelo cargado exitosamente." << endl;
    }

    cout << "\n==================================================" << endl;
    cout << "   INTERFAZ CETZ-LM (Modo Inferencia)" << endl;
    cout << "==================================================\n" << endl;

    string entrada;
    while (true) {
        cout << "Prompt para Cetz > ";
        if (!getline(cin, entrada) || entrada == "salir") break;
        if (entrada.empty()) continue;

        vector<size_t> contexto = tokenizador.codificar(entrada);
        cout << "Cetz-LM -> ";

        for (int step = 0; step < 40; ++step) {
            size_t idSiguiente = cetzModelo->predecirSiguiente(contexto, temperatura, topK, topP, penalizacion);

            // Si el modelo predice <EOS> inmediatamente al inicio (ej. porque el prompt termina en '?' o '.'),
            // no nos detenemos en seco en los primeros 2 tokens para permitir que desarrolle una respuesta.
            if (idSiguiente == tokenizador.idEos) {
                if (step < 2) continue;
                break;
            }

            cout << tokenizador.decodificar(idSiguiente);
            cout.flush();

            contexto.push_back(idSiguiente);
        }
        cout << "\n--------------------------------------------------" << endl;
    }

    return 0;
}
