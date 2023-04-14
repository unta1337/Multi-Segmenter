#include "model.hpp"

int add_vertex(TriangleMesh* mesh, tinyobj::attrib_t& attributes,
              const tinyobj::index_t& idx, std::map<int, int>& known_vertices) {
    if (known_vertices.find(idx.vertex_index) != known_vertices.end())
        return known_vertices[idx.vertex_index];

    const glm::vec3* vertex_array =
        (const glm::vec3*)attributes.vertices.data();
    const glm::vec3* normal_array = (const glm::vec3*)attributes.normals.data();
    const glm::vec2* texcoord_array =
        (const glm::vec2*)attributes.texcoords.data();

    int new_id = (int)mesh->vertex.size();
    known_vertices[idx.vertex_index] = new_id;

    mesh->vertex.push_back(vertex_array[idx.vertex_index]);

    return new_id;
}

void Model::read_obj(std::string obj_file_path) {
    const std::string model_dir =
        obj_file_path.substr(0, obj_file_path.rfind('/') + 1);

    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err = "";

    bool read_ok = tinyobj::LoadObj(&attributes, &shapes, &materials, &err, &err,
                                   obj_file_path.c_str(), model_dir.c_str(),
                                   /* triangulate */ true);
    if (!read_ok) {
        throw std::runtime_error("Could not read OBJ model from " +
                                 obj_file_path + " : " + err);
    }

    if (materials.empty())
        throw std::runtime_error("could not parse materials ...");

    std::cout << "Done loading obj file - found " << shapes.size()
              << " shapes with " << materials.size() << " materials"
              << std::endl;
    std::map<std::string, int> known_textures;

    for (int shape_id = 0; shape_id < (int)shapes.size(); shape_id++) {
        tinyobj::shape_t& shape = shapes[shape_id];

        std::set<int> material_ids;
        for (auto face_mat_id : shape.mesh.material_ids)
            material_ids.insert(face_mat_id);

        std::map<int, int> known_vertices;

        for (int material_id : material_ids) {
            TriangleMesh* mesh = new TriangleMesh;
            Material* mater = new Material;
            mesh->name = shape.name;
            mesh->material = mater;

            for (int face_id = 0; face_id < shape.mesh.material_ids.size();
                 face_id++) {
                if (shape.mesh.material_ids[face_id] != material_id)
                    continue;
                tinyobj::index_t idx0 = shape.mesh.indices[3 * face_id + 0];
                tinyobj::index_t idx1 = shape.mesh.indices[3 * face_id + 1];
                tinyobj::index_t idx2 = shape.mesh.indices[3 * face_id + 2];

                glm::vec3 idx(
                    add_vertex(mesh, attributes, idx0, known_vertices),
                    add_vertex(mesh, attributes, idx1, known_vertices),
                    add_vertex(mesh, attributes, idx2, known_vertices));
                mesh->index.push_back(glm::ivec3(idx.x, idx.y, idx.z));

                mesh->material->diffuse =
                    (const glm::vec3&)materials[material_id].diffuse;
                mesh->material->ambient =
                    (const glm::vec3&)materials[material_id].ambient;
                mesh->material->specular =
                    (const glm::vec3&)materials[material_id].specular;
                mesh->material->emission =
                    (const glm::vec3&)materials[material_id].emission;
                mesh->material->shininess =
                    (const float&)materials[material_id].shininess;
                mesh->material->dissolve =
                    (const float&)materials[material_id].dissolve;
                mesh->material->illumination_model =
                    (const int&)materials[material_id].illum;
                mesh->material->name = materials[material_id].name;
            }

            if (mesh->vertex.empty())
                delete mesh;
            else
                meshes.push_back(mesh);
        }
    }
}
