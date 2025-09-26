#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "tiny_obj_loader.h"

#define TINYOBJLOADER_IMPLEMENTATION

#include <filesystem>
namespace fs = std::filesystem;


using namespace std;
using json = nlohmann::json;

Triangle::Triangle(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, int idx)
    : pos{ p1,p2,p3 }, index_in_mesh(idx) {}

Vertex::Vertex()
    : m_pos(glm::vec3(0)), m_normal(glm::vec3(0)) {}

Vertex::Vertex(glm::vec3 p, glm::vec3 n, glm::vec2 uv)
        : m_pos(p), m_normal(n), m_uv(uv) {}

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadOBJ(const std::string filename)
{
    std::filesystem::path file = std::filesystem::current_path().parent_path() / "scenes" / "objs" / filename;
    std::string filepath = file.string();

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes; 
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;


    bool result = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str());

     // shapes.size() > 0 temporarily ignores problems with materials
    
    if (result)
    {
        int nextTriangleIndex = 0;
        for (size_t s = 0; s < shapes.size(); s++) {
            // Loop over faces(polygon)
            size_t index_offset = 0;
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
                size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

                for (size_t v = 0; v < fv; v++) {

                    // Define new Vertex
                    Vertex newVert;
                    float scale = 3.0;

                    tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                    tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0] * scale;
                    tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1] * scale;
                    tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2] * scale;

                    newVert.m_pos = glm::vec3(vx, vy, vz);

                    if (idx.normal_index >= 0) {
                        tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0] * scale;
                        tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1] * scale;
                        tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2] * scale;

                        // Set vertex position
                        newVert.m_normal = glm::vec3(nx, ny, nz);
                    }

                    /*
                    if (idx.texcoord_index >= 0) {
                        tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                        tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];

                        newVert.m_uv = glm::vec2(tx, ty);
                    }*/


                    // push_back new vertex to buffer
                    this->vertices.push_back(newVert);
                }

                index_offset += fv;

            }
        }
    } 
}


void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
			newMaterial.hasReflective = 1.f;
            newMaterial.hasRefractive = 1.f;
            
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "mesh") {
            newGeom.type = MESH;
            // call load obj here

            // Replace this line:
            // newGeom.filename = p["FILE"]

            // With this code:
            std::string fileStr = p["FILE"];
            newGeom.filename = new char[fileStr.size() + 1];
            std::strcpy(newGeom.filename, fileStr.c_str());

            loadOBJ(fileStr);
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}



