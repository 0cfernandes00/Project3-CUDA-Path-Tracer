#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(std::string filename);
    int loadOBJ(const std::string filename, glm::mat4 transform);
    void loadTexture(const std::string filename);
    void BuildBVH(int count);
    void UpdateNodeBounds(unsigned int nodeIdx);
    void Subdivide(unsigned int nodeIdx);


    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Vertex> vertices;
    std::vector<Triangle> triangles;
    std::vector<int> tri_indices;
    std::vector<Texture> textures;
    std::vector<glm::vec4> texels;
    RenderState state;

    //BVH
    std::vector<BVHNode> bvhTree;
    unsigned int nodesUsed = 1;

};
