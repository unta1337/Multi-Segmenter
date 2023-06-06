#ifndef __OBJLOADER_H
#define __OBJLOADER_H

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "object.h"
#include "vectex.h"

Object load_obj(const std::string& file_path) {
    std::ifstream file(file_path);

    std::vector<Vertex> vertices;
    std::vector<Face> faces;

    if (file.is_open()) {
        std::string line, token;

        while (std::getline(file, line)) {
            std::stringstream line_ss(line);
            std::getline(line_ss, token, ' ');

            if (token == "v") {
                vertices.push_back({
                    (std::getline(line_ss, token, ' '), std::stof(token)),
                    (std::getline(line_ss, token, ' '), std::stof(token)),
                    (std::getline(line_ss, token, ' '), std::stof(token))
                });
            } else if (token == "f") {
                std::string inner_token;
                std::stringstream token_ss;

                if (line_ss.str().find('/') != std::string::npos) {
                    faces.push_back({
                        (std::getline(line_ss, token, ' '), token_ss.str(token), std::getline(token_ss, inner_token, '/'), std::stoull(inner_token)) - 1,
                        (std::getline(line_ss, token, ' '), token_ss.str(token), std::getline(token_ss, inner_token, '/'), std::stoull(inner_token)) - 1,
                        (std::getline(line_ss, token, ' '), token_ss.str(token), std::getline(token_ss, inner_token, '/'), std::stoull(inner_token)) - 1
                    });
                } else {
                    faces.push_back({
                        (std::getline(line_ss, token, ' '), std::stoull(token)) - 1,
                        (std::getline(line_ss, token, ' '), std::stoull(token)) - 1,
                        (std::getline(line_ss, token, ' '), std::stoull(token)) - 1
                    });
                }
            }
        }
        file.close();
    }

    return { vertices, faces };
}

#endif
