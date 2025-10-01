#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>
#include <array>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
	MESH
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};


struct Vertex
{
    glm::vec3 m_pos;
    glm::vec3 m_normal;
    glm::vec2 m_uv;

    /*    glm::vec3 m_color; */

    Vertex();
    Vertex(glm::vec3 p, glm::vec3 nor, glm::vec2 uv);
};

class Triangle {
public:
    Vertex v1;
    Vertex v2;
    Vertex v3;
    glm::vec3 centroid;
    int index_in_mesh; // What index in the mesh's vector<Triangles> does this sit at?

    Triangle();
    Triangle(Vertex p1, Vertex p2, Vertex p3, int idx);
    
    glm::vec3 operator[](unsigned int) const;

    //friend class Mesh;
};

struct BVHNode 
{
    glm::vec3 aabbMin, aabbMax;
    unsigned int leftChild, rightChild;
    __host__ __device__ bool isLeaf() const {
        return primCount > 0;
    };
    unsigned int firstPrim, primCount;
};


class Texture {
public:
    float width;
    float height;
    int startPixelTex;
    
    //unsigned char* pixelData;
};

/*
class Texture2D : public Texture {
public:
    Texture2D(OpenGLContext* context, int arrayIdx, int associatedTexSlot);
    ~Texture2D();

    void create(const char* texturePath, bool wrap) override;
    void create(bool wrap) override;
};

struct Texture {
    float texels[];

};
*/


struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    char* filename;
    Vertex* verts;
    Triangle* triangle;

};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
    int diffuseTextureID;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    float lensRadius = 0.05f;
    float focalDistance = 5.0f;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  glm::vec2 uv;
  glm::vec2 surfaceUVCoord;
};
