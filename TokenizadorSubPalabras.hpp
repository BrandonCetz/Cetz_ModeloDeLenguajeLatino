#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>

class TokenizadorSubPalabras {
private:
    // Guarda el orden de fusiones aprendidas: ("C", "O") -> "CO"
    std::vector<std::pair<std::string, std::string>> reglasFusion_;
    // Mapa de rango rÃ¡pido para codificaciÃ³n: ("A", "B") -> prioridad
    std::map<std::pair<std::string, std::string>, int> rangoFusion_;

public:
    std::unordered_map<std::string, size_t> tokenAId;
    std::unordered_map<size_t, std::string> idAToken;

    size_t idPad = 0;
    size_t idUnk = 1;
    size_t idEos = 2;

    TokenizadorSubPalabras() {
        inicializarEspeciales();
    }

    void inicializarEspeciales() {
        tokenAId.clear();
        idAToken.clear();
        tokenAId["<PAD>"] = idPad; idAToken[idPad] = "<PAD>";
        tokenAId["<UNK>"] = idUnk; idAToken[idUnk] = "<UNK>";
        tokenAId["<EOS>"] = idEos; idAToken[idEos] = "<EOS>";
    }

    // Divide el texto en unidades bÃ¡sicas preservando signos de puntuaciÃ³n y codificaciÃ³n UTF-8
    std::vector<std::string> separarPalabras(const std::string& texto) const {
        std::vector<std::string> unidades;
        std::string buffer = "";

        auto esEspacio = [](unsigned char c) {
            return c == ' ' || c == '\t' || c == '\r' || c == '\n';
        };

        auto esPuntuacion = [](const std::string& s) {
            return s == "." || s == "," || s == ";" || s == ":" ||
                   s == "?" || s == "Â¿" || s == "!" || s == "Â¡" ||
                   s == "(" || s == ")" || s == "\"" || s == "-" ||
                   s == "Â«" || s == "Â»" || s == "â€”";
        };

        size_t i = 0;
        while (i < texto.size()) {
            unsigned char c = static_cast<unsigned char>(texto[i]);

            if (esEspacio(c)) {
                if (!buffer.empty()) {
                    unidades.push_back(buffer + "_");
                    buffer.clear();
                }
                i++;
                continue;
            }

            // Detectar la longitud del carÃ¡cter UTF-8
            size_t bytesChar = 1;
            if ((c & 0x80) == 0x00) bytesChar = 1;        // ASCII estÃ¡ndar (1 byte)
            else if ((c & 0xE0) == 0xC0) bytesChar = 2;   // Ã±, Ã¡, Ã©, Ã­, Ã³, Ãº, Â¿, Â¡ (2 bytes)
            else if ((c & 0xF0) == 0xE0) bytesChar = 3;   // Caracteres especiales (3 bytes)
            else if ((c & 0xF8) == 0xF0) bytesChar = 4;   // Emojis / sÃ­mbolos (4 bytes)

            if (i + bytesChar > texto.size()) bytesChar = 1;
            std::string sub = texto.substr(i, bytesChar);
            i += bytesChar;

            if (esPuntuacion(sub)) {
                if (!buffer.empty()) {
                    unidades.push_back(buffer + "_");
                    buffer.clear();
                }
                unidades.push_back(sub + "_");
            } else {
                buffer += sub;
            }
        }

        if (!buffer.empty()) {
            unidades.push_back(buffer + "_");
        }

        return unidades;
    }

    // Algoritmo de entrenamiento BPE respetando UTF-8 y caracteres del espaÃ±ol latino
    void construirVocabulario(const std::vector<std::string>& dataset, size_t tamanoDeseado) {
        inicializarEspeciales();
        reglasFusion_.clear();
        rangoFusion_.clear();

        size_t idActual = 3;

        // 1. Contar frecuencia de cada palabra/unidad
        std::unordered_map<std::string, int> frecuencias;
        for (const auto& frase : dataset) {
            auto palabras = separarPalabras(frase);
            for (const auto& p : palabras) frecuencias[p]++;
        }

        // 2. Descomponer cada palabra en sus caracteres UTF-8 base
        std::unordered_map<std::string, std::vector<std::string>> divisiones;
        for (const auto& par : frecuencias) {
            std::vector<std::string> caracteres;
            const std::string& str = par.first;
            size_t i = 0;
            while (i < str.size()) {
                unsigned char c = static_cast<unsigned char>(str[i]);
                size_t len = 1;
                if ((c & 0x80) == 0x00) len = 1;
                else if ((c & 0xE0) == 0xC0) len = 2;
                else if ((c & 0xF0) == 0xE0) len = 3;
                else if ((c & 0xF8) == 0xF0) len = 4;
                if (i + len > str.size()) len = 1;

                std::string ch = str.substr(i, len);
                caracteres.push_back(ch);
                i += len;

                if (tokenAId.find(ch) == tokenAId.end()) {
                    tokenAId[ch] = idActual;
                    idAToken[idActual] = ch;
                    idActual++;
                }
            }
            divisiones[par.first] = caracteres;
        }

        // 3. Aprender fusiones (Merges)
        int orden = 0;
        while (idActual < tamanoDeseado) {
            std::map<std::pair<std::string, std::string>, int> frecPares;
            for (const auto& par : frecuencias) {
                const auto& divs = divisiones[par.first];
                if (divs.size() < 2) continue;
                for (size_t i = 0; i < divs.size() - 1; ++i) {
                    frecPares[{divs[i], divs[i + 1]}] += par.second;
                }
            }

            if (frecPares.empty()) break;

            auto mejorPar = frecPares.begin();
            for (auto it = frecPares.begin(); it != frecPares.end(); ++it) {
                if (it->second > mejorPar->second) mejorPar = it;
            }

            if (mejorPar->second < 2) break; // Detener si las fusiones ya no se repiten

            std::string a = mejorPar->first.first;
            std::string b = mejorPar->first.second;
            std::string nuevoToken = a + b;

            reglasFusion_.push_back({a, b});
            rangoFusion_[{a, b}] = orden++;

            tokenAId[nuevoToken] = idActual;
            idAToken[idActual] = nuevoToken;
            idActual++;

            // Aplicar fusiÃ³n a las divisiones actuales
            for (auto& par : divisiones) {
                std::vector<std::string> nuevaDiv;
                const auto& divs = par.second;
                for (size_t i = 0; i < divs.size(); ++i) {
                    if (i < divs.size() - 1 && divs[i] == a && divs[i + 1] == b) {
                        nuevaDiv.push_back(nuevoToken);
                        i++;
                    } else {
                        nuevaDiv.push_back(divs[i]);
                    }
                }
                divisiones[par.first] = nuevaDiv;
            }
        }
    }

    size_t vocabSize() const { return idAToken.size(); }

    // Codifica texto a secuencia de IDs de tokens
    std::vector<size_t> codificar(const std::string& texto) const {
        auto palabras = separarPalabras(texto);
        std::vector<size_t> ids;

        for (const auto& palabra : palabras) {
            std::vector<std::string> divs;
            size_t i = 0;
            while (i < palabra.size()) {
                unsigned char c = static_cast<unsigned char>(palabra[i]);
                size_t len = 1;
                if ((c & 0x80) == 0x00) len = 1;
                else if ((c & 0xE0) == 0xC0) len = 2;
                else if ((c & 0xF0) == 0xE0) len = 3;
                else if ((c & 0xF8) == 0xF0) len = 4;
                if (i + len > palabra.size()) len = 1;

                divs.push_back(palabra.substr(i, len));
                i += len;
            }

            for (const auto& regla : reglasFusion_) {
                std::vector<std::string> nuevaDiv;
                for (size_t j = 0; j < divs.size(); ++j) {
                    if (j < divs.size() - 1 && divs[j] == regla.first && divs[j + 1] == regla.second) {
                        nuevaDiv.push_back(regla.first + regla.second);
                        j++;
                    } else {
                        nuevaDiv.push_back(divs[j]);
                    }
                }
                divs = nuevaDiv;
            }

            for (const auto& token : divs) {
                auto it = tokenAId.find(token);
                if (it != tokenAId.end()) {
                    ids.push_back(it->second);
                } else {
                    ids.push_back(idUnk);
                }
            }
        }
        return ids;
    }

    // Decodifica un ID a string
    std::string decodificar(size_t id) const {
        auto it = idAToken.find(id);
        if (it != idAToken.end()) {
            std::string token = it->second;
            size_t pos = token.find('_');
            if (pos != std::string::npos) {
                token.replace(pos, 1, " ");
            }
            return token;
        }
        return "<UNK>";
    }

    // Decodifica un vector completo de IDs a string reconstruido
    std::string decodificar(const std::vector<size_t>& ids) const {
        std::string texto = "";
        for (size_t id : ids) {
            if (id == idEos) break;
            if (id == idPad) continue;
            texto += decodificar(id);
        }
        return texto;
    }

    // Guardar vocabulario y reglas ordenadas a disco (formato TSV robusto)
    bool guardar(const std::string& ruta) const {
        std::ofstream out(ruta);
        if (!out.is_open()) return false;

        out << idAToken.size() << "\n";
        for (const auto& par : idAToken) {
            out << par.first << "\t" << par.second << "\n";
        }

        out << reglasFusion_.size() << "\n";
        for (const auto& regla : reglasFusion_) {
            out << regla.first << "\t" << regla.second << "\n";
        }

        return true;
    }

    // Cargar vocabulario y reglas desde disco
    bool cargar(const std::string& ruta) {
        std::ifstream in(ruta);
        if (!in.is_open()) return false;

        inicializarEspeciales();
        reglasFusion_.clear();
        rangoFusion_.clear();

        std::string linea;
        if (!std::getline(in, linea)) return false;
        size_t totalTokens = std::stoull(linea);

        for (size_t i = 0; i < totalTokens; ++i) {
            if (!std::getline(in, linea)) return false;
            size_t tabPos = linea.find('\t');
            if (tabPos == std::string::npos) continue;
            size_t id = std::stoull(linea.substr(0, tabPos));
            std::string token = linea.substr(tabPos + 1);
            tokenAId[token] = id;
            idAToken[id] = token;
        }

        if (!std::getline(in, linea)) return true;
        size_t totalReglas = std::stoull(linea);

        for (size_t i = 0; i < totalReglas; ++i) {
            if (!std::getline(in, linea)) break;
            size_t tabPos = linea.find('\t');
            if (tabPos == std::string::npos) continue;
            std::string a = linea.substr(0, tabPos);
            std::string b = linea.substr(tabPos + 1);
            reglasFusion_.push_back({a, b});
            rangoFusion_[{a, b}] = static_cast<int>(i);
        }

        return true;
    }
};
