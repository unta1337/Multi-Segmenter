#include "Model.h"
#include <set>
#include <iostream>
#include <tiny_obj_loader.h>

int addVertex(TriangleMesh* mesh,
	tinyobj::attrib_t& attributes,
	const tinyobj::index_t& idx,
	std::map<int, int>& knownVertices)
{
	if (knownVertices.find(idx.vertex_index) != knownVertices.end())
		return knownVertices[idx.vertex_index];

	const glm::vec3* vertex_array = (const glm::vec3*)attributes.vertices.data();
	const glm::vec3* normal_array = (const glm::vec3*)attributes.normals.data();
	const glm::vec2* texcoord_array = (const glm::vec2*)attributes.texcoords.data();

	int newID = (int)mesh->vertex.size();
	knownVertices[idx.vertex_index] = newID;

	mesh->vertex.push_back(vertex_array[idx.vertex_index]);

	return newID;
}

void Model::readOBJ(std::string objFilePath) {
	const std::string modelDir
		= objFilePath.substr(0, objFilePath.rfind('/') + 1);

	tinyobj::attrib_t attributes;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err = "";

	bool readOK
		= tinyobj::LoadObj(&attributes,
			&shapes,
			&materials,
			&err,
			&err,
			objFilePath.c_str(),
			modelDir.c_str(),
			/* triangulate */true);
	if (!readOK) {
		throw std::runtime_error("Could not read OBJ model from " + objFilePath + " : " + err);
	}

	if (materials.empty())
		throw std::runtime_error("could not parse materials ...");

	std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;
	std::map<std::string, int>      knownTextures;

	for (int shapeID = 0; shapeID < (int)shapes.size(); shapeID++) {
		tinyobj::shape_t& shape = shapes[shapeID];

		std::set<int> materialIDs;
		for (auto faceMatID : shape.mesh.material_ids)
			materialIDs.insert(faceMatID);

		std::map<int, int> knownVertices;

		for (int materialID : materialIDs) {
			TriangleMesh* mesh = new TriangleMesh;
			Material_t* mater = new Material_t;
			mesh->name = shape.name;
			mesh->material = mater;

			for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++) {
				if (shape.mesh.material_ids[faceID] != materialID) continue;
				tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
				tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
				tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

				glm::vec3 idx(addVertex(mesh, attributes, idx0, knownVertices),
					addVertex(mesh, attributes, idx1, knownVertices),
					addVertex(mesh, attributes, idx2, knownVertices));
				mesh->index.push_back(glm::ivec3(idx.x, idx.y, idx.z));

				mesh->material->diffuse = (const glm::vec3&)materials[materialID].diffuse;
				mesh->material->ambient = (const glm::vec3&)materials[materialID].ambient;
				mesh->material->specular = (const glm::vec3&)materials[materialID].specular;
				mesh->material->emmision = (const glm::vec3&)materials[materialID].emission;
				mesh->material->shininess = (const float&)materials[materialID].shininess;
				mesh->material->dissolve = (const float&)materials[materialID].dissolve;
				mesh->material->illuminationModel = (const int&)materials[materialID].illum;
				mesh->material->name = materials[materialID].name;
			}

			if (mesh->vertex.empty())
				delete mesh;
			else
				meshes.push_back(mesh);
		}
	}
}