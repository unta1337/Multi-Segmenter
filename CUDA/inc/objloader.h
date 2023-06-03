#ifndef __OBJLOADER_H
#define __OBJLOADER_H

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "dtypes.h"

object_t load_obj(std::string file_path) {
    std::ifstream file(file_path);

    std::vector<vertex_t> vertices;
    std::vector<face_t> faces;

    vertices.push_back({});     // .obj 파일의 1-based 인덱싱을 위한 더미 정점.

    if (file.is_open()) {
        std::string line, token;

        while (std::getline(file, line)) {
            std::stringstream line_ss(line);
            std::getline(line_ss, token, ' ');

            if (token.compare("v") == 0) {
                vertices.push_back({
                    (std::getline(line_ss, token, ' '), std::stof(token)),
                    (std::getline(line_ss, token, ' '), std::stof(token)),
                    (std::getline(line_ss, token, ' '), std::stof(token))
                });
            } else if (token.compare("f") == 0) {
                std::string inner_token;
                std::stringstream token_ss;

                faces.push_back({
                    (std::getline(line_ss, token, ' '), token_ss.str(token), std::getline(token_ss, inner_token, '/'), std::stoull(inner_token)),
                    (std::getline(line_ss, token, ' '), token_ss.str(token), std::getline(token_ss, inner_token, '/'), std::stoull(inner_token)),
                    (std::getline(line_ss, token, ' '), token_ss.str(token), std::getline(token_ss, inner_token, '/'), std::stoull(inner_token))
                });
            }
        }
        file.close();
    }

    return { vertices, faces };
}

#endif
