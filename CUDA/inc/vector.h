#ifndef __VECTOR_H
#define __VECTOR_H

#include <functional> // std::hash

struct Vector3f {
    float x;
    float y;
    float z;
};

struct Vector2f { // for textCoord
    float x;
    float y;
};

struct Vector4f { // for color.hpp
    float x;
    float y;
    float z;
    float w;
};

struct Vector3u {
    size_t x;
    size_t y;
    size_t z;
};

namespace std {
    template<>
    struct hash<Vector3f> {
        std::size_t operator()(const Vector3f& vector) const {
            // Use the std::hash function to hash the individual components of the
            // vector
            std::size_t h1 = std::hash<float>()(vector.x);
            std::size_t h2 = std::hash<float>()(vector.y);
            std::size_t h3 = std::hash<float>()(vector.z);

            // Combine the individual component hashes into a single hash value
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
}

#endif