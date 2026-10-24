#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <cctype>

struct TokenizadorPalabras {
    std::unordered_map<std::string, size_t> wordToId;
    std::unordered_map<size_t, std::string> idToWord;

    size_t padId;
    size_t unkId;
    size_t eosId;

    std::string limpiarPalabra(const std::string& word) const {
        std::string limpia = "";
        for (char c : word) {
            if (std::isalnum(static_cast<unsigned char>(c)) || c == '\'' || c == 'ñ' || c == 'Ñ') {
                limpia += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            }
        }
        return limpia;
    }

    std::vector<std::string> separarPalabras(const std::string& texto) const {
        std::stringstream ss(texto);
        std::string token;
        std::vector<std::string> palabras;
        while (ss >> token) {
            std::string limpia = limpiarPalabra(token);
            if (!limpia.empty()) {
                palabras.push_back(limpia);
            }
        }
        return palabras;
    }

    void construirVocabulario(const std::vector<std::string>& dataset) {
        wordToId.clear();
        idToWord.clear();

        // Tokens especiales del estándar Transformer
        padId = 0; wordToId["<PAD>"] = padId; idToWord[padId] = "<PAD>";
        unkId = 1; wordToId["<UNK>"] = unkId; idToWord[unkId] = "<UNK>";
        eosId = 2; wordToId["<EOS>"] = eosId; idToWord[eosId] = "<EOS>";

        size_t idActual = 3;
        for (const auto& frase : dataset) {
            auto palabras = separarPalabras(frase);
            for (const auto& p : palabras) {
                if (wordToId.find(p) == wordToId.end()) {
                    wordToId[p] = idActual;
                    idToWord[idActual] = p;
                    idActual++;
                }
            }
        }
    }

    size_t vocabSize() const { return idToWord.size(); }

    std::vector<size_t> codificar(const std::string& texto) const {
        auto palabras = separarPalabras(texto);
        std::vector<size_t> ids;
        ids.reserve(palabras.size());
        for (const auto& p : palabras) {
            if (wordToId.count(p)) {
                ids.push_back(wordToId.at(p));
            } else {
                ids.push_back(unkId); // Asigna <UNK> si no conoce la palabra
            }
        }
        return ids;
    }

    std::string decodificar(size_t id) const {
        if (idToWord.count(id)) return idToWord.at(id);
        return "<UNK>";
    }
};
