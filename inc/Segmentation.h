#pragma once
#include "Model.h"
#include <algorithm>
#include <chrono>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/normal.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <iostream>
#include <stack>
#include <unordered_map>

namespace FaceGraph {
typedef struct Triangle_t {
    glm::vec3 vert[3];
} Triangle_t;

bool isConnected(Triangle_t a, Triangle_t b) {
    int v = 0;
    if (glm::all(glm::equal(a.vert[0], b.vert[0])) ||
        glm::all(glm::equal(a.vert[0], b.vert[1])) ||
        glm::all(glm::equal(a.vert[0], b.vert[2])))
        v++;
    if (glm::all(glm::equal(a.vert[1], b.vert[0])) ||
        glm::all(glm::equal(a.vert[1], b.vert[1])) ||
        glm::all(glm::equal(a.vert[1], b.vert[2])))
        v++;
    if (glm::all(glm::equal(a.vert[2], b.vert[0])) ||
        glm::all(glm::equal(a.vert[2], b.vert[1])) ||
        glm::all(glm::equal(a.vert[2], b.vert[2])))
        v++;

    return (v > 1);
}

class graphNode_t {
  public:
    int triangles;
    graphNode_t* link;

    ~graphNode_t() {
        delete (link);
    }
};

struct Vec3Hash {
    std::size_t operator()(const glm::vec3& v) const {
        // Use the std::hash function to hash the individual components of the
        // vector
        std::size_t h1 = std::hash<float>()(v.x);
        std::size_t h2 = std::hash<float>()(v.y);
        std::size_t h3 = std::hash<float>()(v.z);

        // Combine the individual component hashes into a single hash value
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

class FaceGraph {
  public:
    std::vector<std::vector<int>> adj_list;
    std::vector<Triangle_t>* ref_vector;

    FaceGraph() {
    }
    FaceGraph(std::vector<Triangle_t>* _list) {
        ref_vector = _list;

        std::unordered_map<glm::vec3, std::vector<int>, Vec3Hash> vertex_map;
        for (int i = 0; i < _list->size(); i++) {
            for (int j = 0; j < 3; j++) {
                glm::vec3 vertex = _list->at(i).vert[j];
                vertex_map[vertex].push_back(i);
            }
        }

        adj_list = std::vector<std::vector<int>>(_list->size());
        for (int i = 0; i < _list->size(); i++) {
            for (int j = 0; j < 3; j++) {
                glm::vec3 vertex = _list->at(i).vert[j];
                std::vector<int> adjacent_triangles = vertex_map[vertex];
                for (int k = 0; k < adjacent_triangles.size(); k++) {
                    int adjacent_triangle = adjacent_triangles[k];
                    if (i != adjacent_triangle &&
                        isConnected(_list->at(i),
                                    _list->at(adjacent_triangle))) {
                        adj_list[i].push_back(adjacent_triangle);
                    }
                }
            }
        }
    }

    std::vector<std::vector<Triangle_t>> check_connected() {
        std::vector<int> isVisit(adj_list.size());

        int count = 0;
        for (int i = 0; i < adj_list.size(); i++) {
            if (isVisit[i] == 0) {
                count++;
                traverse_dfs(&isVisit, i, count);
                // std::cout << "// ";
            }
        }

        // std::cout << "Component number : " << count<< std::endl;

        std::vector<std::vector<Triangle_t>> componnentList(count);

        for (int i = 0; i < isVisit.size(); i++) {
            componnentList[isVisit[i] - 1].push_back(ref_vector->data()[i]);
        }

        return componnentList;
    }

    void traverse_dfs(std::vector<int>* visit, int startVert, int count) {
        // std::cout << startVert << " ";
        std::stack<int> dfsStack;
        dfsStack.push(startVert);

        while (!dfsStack.empty()) {
            int currentVert = dfsStack.top();
            dfsStack.pop();

            visit->data()[currentVert] = count;
            for (int i = 0; i < adj_list[currentVert].size(); i++) {
                int adjacent_triangle = adj_list[currentVert][i];
                if (visit->data()[adjacent_triangle] == 0) {
                    dfsStack.push(adjacent_triangle);
                }
            }
        }
        /*visit->data()[startVert] = count;

        for (int i = 0; i < adj_list[startVert].size(); i++) {
            int adjacent_triangle = adj_list[startVert][i];
            if (visit->data()[adjacent_triangle] == 0) {
                traverse_dfs(visit, adjacent_triangle, count);
            }
        }*/
    }
};

TriangleMesh* TriangleListToObj(std::vector<Triangle_t> __list) {
    TriangleMesh* subObject = new TriangleMesh();

    Material_t* subMTL = new Material_t();

    std::unordered_map<glm::vec3, size_t, Vec3Hash> vertexMap;
    subObject->index.resize(__list.size());
    size_t vertIdx = 1;
    for (int i = 0; i < __list.size(); i++) {
        glm::ivec3 index;
        for (int j = 0; j < 3; j++) {
            auto vertexItem = vertexMap.find(__list[i].vert[j]);

            if (vertexItem != vertexMap.end()) {
                index[j] = vertexItem->second;
            } else {
                vertexMap.insert({__list[i].vert[j], vertIdx});
                index[j] = vertIdx++;
            }
            // auto vertIter = std::find(subObject->vertex.begin(),
            // subObject->vertex.end(), __list[i].vert[j]);

            ////찾은경우
            // if (vertIter != subObject->vertex.end()) {
            //     index[j] = vertIter - subObject->vertex.begin() + 1;
            // }
            ////못찾은경우
            // else {
            //     subObject->vertex.push_back(__list[i].vert[j]);
            //     index[j] = subObject->vertex.end() -
            //     subObject->vertex.begin();
            // }
        }
        subObject->index[i] = index;
    }

    subObject->vertex.resize(vertIdx);
    for (auto vItem : vertexMap) {
        subObject->vertex[vItem.second - 1] = vItem.first;
    }

    subObject->material = subMTL;

    return subObject;
}
} // namespace FaceGraph

namespace Color {
glm::vec4 getColorfromJET(float v, float vmin, float vmax) {
    glm::vec3 c = {1.0, 1.0, 1.0};
    float dv;

    if (v < vmin)
        v = vmin;
    if (v > vmax)
        v = vmax;
    dv = vmax - vmin;

    if (v < (vmin + 0.25 * dv)) {
        c.r = 0;
        c.g = 4 * (v - vmin) / dv;
    } else if (v < (vmin + 0.5 * dv)) {
        c.r = 0;
        c.b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
    } else if (v < (vmin + 0.75 * dv)) {
        c.r = 4 * (v - vmin - 0.5 * dv) / dv;
        c.b = 0;
    } else {
        c.g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
        c.b = 0;
    }
    return glm::vec4(c, 1.0f);
}
} // namespace Color

class Segmentation_t {
  public:
    TriangleMesh* mesh;
    float tolerance;

    Segmentation_t() : tolerance(0.0f) {
    }
    Segmentation_t(TriangleMesh* _mesh, float _tolerance)
        : mesh(_mesh), tolerance(_tolerance) {
    }

    std::vector<TriangleMesh*> DoSegmentation() {
        std::vector<glm::vec3> facenormals(mesh->index.size());

        for (int i = 0; i < mesh->index.size(); i++) {
            facenormals[i] = glm::triangleNormal(
                mesh->vertex[mesh->index[i].x], mesh->vertex[mesh->index[i].y],
                mesh->vertex[mesh->index[i].z]);
        }

        std::cout << "Normal vector compute done" << std::endl;

        int count = facenormals.size();

        std::unordered_map<glm::vec3, size_t, FaceGraph::Vec3Hash> countMap;

        for (int i = 0; i < count; i++) {
            glm::vec3 targetNorm = facenormals[i];

            for (auto iter : countMap) {
                glm::vec3 _compare = iter.first;
                float normAngle =
                    glm::degrees(glm::angle(_compare, targetNorm));

                if (normAngle < tolerance) {
                    targetNorm = _compare;
                    break;
                }
            }

            auto item = countMap.find(targetNorm);

            if (item == countMap.end()) {
                countMap.insert({targetNorm, 0});
            }
            countMap[targetNorm]++;
        }

        std::unordered_map<glm::vec3, std::vector<FaceGraph::Triangle_t>,
                           FaceGraph::Vec3Hash>
            myMap;
        for (auto& iter : countMap) {
            // std::cout << "map["<< glm::to_string(iter.first) <<"] : " <<
            // iter.second << std::endl;
            myMap.insert(
                {iter.first, std::vector<FaceGraph::Triangle_t>(iter.second)});
            // std::cout << "check : " << myMap[iter.first].size() << std::endl;
            iter.second = 0;
        }
        std::cout << "map count complete (map size : " << countMap.size() << ")"
                  << std::endl;

        for (int i = 0; i < count; i++) {
            glm::vec3 targetNorm = facenormals[i];

            for (auto& iter : countMap) {
                glm::vec3 _compare = iter.first;
                float normAngle =
                    glm::degrees(glm::angle(_compare, targetNorm));

                if (normAngle < tolerance) {
                    targetNorm = _compare;
                    break;
                }
            }

            auto indexs = countMap.find(targetNorm);

            indexs->second++;
        }

        for (auto& iter : countMap) {
            if (myMap[iter.first].size() != iter.second) {
                // std::cout << "Validation steps: " << std::endl;
                // std::cout << "map[" << glm::to_string(iter.first) << "] : "
                // << iter.second << std::endl; std::cout << "check : " <<
                // myMap[iter.first].size() << std::endl;
                myMap[iter.first].resize(iter.second);
            }

            if (iter.second == 0) {
                std::cout << "Removed map : " << glm::to_string(iter.first)
                          << " " << iter.second << std::endl;
                myMap.erase(iter.first);
                countMap.erase(iter.first);
            }

            iter.second = 0;
        }
        std::cout << "map compaction complete (map size : " << countMap.size()
                  << ")" << std::endl;

        double TotalTime = 0.0;
        for (int i = 0; i < count; i++) {
            glm::vec3 targetNorm = facenormals[i];

            for (auto& iter : countMap) {
                glm::vec3 _compare = iter.first;
                float normAngle =
                    glm::degrees(glm::angle(_compare, targetNorm));

                if (normAngle < tolerance) {
                    targetNorm = _compare;
                    break;
                }
            }

            auto item = myMap.find(targetNorm);
            auto indexs = countMap.find(targetNorm);

            FaceGraph::Triangle_t tri;
            tri.vert[0] = mesh->vertex[mesh->index[i].x];
            tri.vert[1] = mesh->vertex[mesh->index[i].y];
            tri.vert[2] = mesh->vertex[mesh->index[i].z];

            item->second[indexs->second++] = tri;
        }

        std::cout << "Normal map insert done total (" << myMap.size()
                  << ") size map" << std::endl;

        // std::cout << "map size : " << myMap.size() << std::endl;

        std::vector<TriangleMesh*> result;
        int number = 0;
        for (auto iter : myMap) {
            auto startTime = std::chrono::system_clock::now();
            // std::cout << "Key[" << iter.first.x << ", " << iter.first.y << ",
            // " << iter.first.z << "] : "; std::cout << "Number of faces : " <<
            // iter.second.size() << std::endl;

            FaceGraph::FaceGraph fg(&iter.second);
            std::cout << "Face Graph done" << std::endl;
            std::vector<std::vector<FaceGraph::Triangle_t>> temp =
                fg.check_connected();
            std::cout << "Check connected done" << std::endl;

            for (auto subs : temp) {
                TriangleMesh* subObject = FaceGraph::TriangleListToObj(subs);
                subObject->material->diffuse = glm::vec3(1, 0, 0);
                subObject->material->name =
                    "sub_materials_" + std::to_string(number);
                subObject->name =
                    mesh->name + "_seg_" + std::to_string(number++);

                result.push_back(subObject);
            }

            auto endTime = std::chrono::system_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::microseconds>(
                          endTime - startTime)
                          .count();

            TotalTime += (ms / 1000.);
            std::cout << "Spend : " << TotalTime << " ms (" << (ms / 1000.)
                      << " ms)" << std::endl;
        }
        std::cout << "Check connectivity and Make triangle mesh done"
                  << std::endl;

        for (int i = 0; i < result.size(); i++) {
            result[i]->material->diffuse =
                Color::getColorfromJET(i, 0, result.size());
            result[i]->material->ambient = glm::vec3(1.0f, 1.0f, 1.0f);
            result[i]->material->specular = glm::vec3(0.5f, 0.5f, 0.5f);
        }

        myMap.clear();

        return result;
    }
};
