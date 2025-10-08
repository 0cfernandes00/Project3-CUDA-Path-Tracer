#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#include <iostream>
#include <OpenImageDenoise/oidn.hpp>

#define ERRORCHECK 1
#define M_PI 3.14159265358979323846
#define RAY_EPSILON 0.00005f
#define EPSILON_RR 0.00005f

#define DENOISE_ITERATION 50

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}


__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

static Vertex* dev_vertices = NULL;
static Triangle* dev_triangles = NULL;
static BVHNode* dev_nodes = NULL;
static int* dev_tri_indices = NULL;
static int* dev_rootNodeIdx = NULL;
static Texture* dev_textures = NULL;
static glm::vec4* dev_texels = NULL;

#if DENOISE
    static glm::vec3* dev_displayImg = NULL; 
    static glm::vec3* dev_denoiseImg = NULL;
    static glm::vec3* dev_albedoImg = NULL;
    static glm::vec3* dev_normalsImg = NULL;
    static oidn::DeviceRef oidn_device;
#endif

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

    cudaMalloc(&dev_vertices, scene->vertices.size() * sizeof(Vertex));
    cudaMemcpy(dev_vertices, scene->vertices.data(), scene->vertices.size() * sizeof(Vertex), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_nodes, scene->bvhTree.size() * sizeof(BVHNode));
    cudaMemcpy(dev_nodes, scene->bvhTree.data(), scene->bvhTree.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_tri_indices, scene->tri_indices.size() * sizeof(int));
    cudaMemcpy(dev_tri_indices, scene->tri_indices.data(), scene->tri_indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_rootNodeIdx, sizeof(int));
    cudaMemset(dev_rootNodeIdx, 0, sizeof(int));

    cudaMalloc(&dev_textures, scene->textures.size() * sizeof(Texture));
    cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_texels, scene->texels.size() * sizeof(glm::vec4));
    cudaMemcpy(dev_texels, scene->texels.data(), scene->texels.size() * sizeof(glm::vec4), cudaMemcpyHostToDevice);

#if DENOISE
    cudaMalloc(&dev_displayImg, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_displayImg, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_denoiseImg, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoiseImg, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_albedoImg, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_albedoImg, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_normalsImg, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_normalsImg, 0, pixelcount * sizeof(glm::vec3));

    OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_CPU);
    oidnCommitDevice(device);
#endif

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created

    cudaFree(dev_vertices);
    cudaFree(dev_triangles);
    cudaFree(dev_nodes);
    cudaFree(dev_tri_indices);
    cudaFree(dev_rootNodeIdx);
    cudaFree(dev_textures);
    cudaFree(dev_texels);

#if DENOISE
    cudaFree(dev_displayImg);
    cudaFree(dev_denoiseImg);
    cudaFree(dev_albedoImg);
    cudaFree(dev_normalsImg);
#endif

    checkCUDAError("pathtraceFree");
}



/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, bool antiAlias, bool dof)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;



    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];


        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, pathSegments[index].remainingBounces);
        thrust::uniform_real_distribution<float> u05(-0.5, 0.5);

        float x_rng = 0.f;
        float y_rng = 0.f;

        if (antiAlias) {
            x_rng = u05(rng);
            y_rng = u05(rng);
        }
        // jitter ray direction for antialiasing

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + x_rng - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + y_rng - (float)cam.resolution.y * 0.5f)
        );

        if (dof) {
            // lens effect Depth of Field      
            if (cam.lensRadius > 0) {
                // Sample point on lens

                // returns a vec3
                    // TODO
                glm::vec2 sample(0, 0);

                thrust::uniform_real_distribution<float> u01(0, 1);
                glm::vec2 xi = glm::vec2(u01(rng), u01(rng));

                glm::vec2 offset = 2.0f * xi - glm::vec2(1, 1);
                if (offset.x == 0 && offset.y == 0) {
                    sample = glm::vec2(0.f);
                }

                float theta, r;
                if (abs(offset.x) > abs(offset.y)) {
                    r = offset.x;
                    theta = PI / 4.f * (offset.y / offset.x);
                }
                else {
                    r = offset.y;
                    theta = PI / 2.f - PI / 4.f * (offset.x / offset.y);
                }
                sample = r * glm::vec2(cos(theta), sin(theta));



                glm::vec2 pLens = cam.lensRadius * sample;
                glm::vec3 pLens_world = pLens.x * cam.right + pLens.y * cam.up;

                // Compute point on plane of focus
                float ft = cam.focalDistance / segment.ray.direction.z;
                glm::vec3 pFocus = cam.position + segment.ray.direction * cam.focalDistance;

                // Update ray for effect of lens 
                segment.ray.origin = cam.position + pLens_world;
                segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);

            }
        }
        

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}


__host__ __device__ glm::vec3 barycentricCoords(
    glm::vec3 p,
    glm::vec3 v0,
    glm::vec3 v1,
    glm::vec3 v2)
{
    glm::vec3 sideA = v1 - v2;
    glm::vec3 sideB = v2 - v0;
    float s = glm::length(glm::cross(sideA, sideB));

    sideA = p - v1;
    sideB = p - v2;
    float s1 = glm::length(glm::cross(sideA, sideB)) / s;

    sideA = p - v0;
    sideB = p - v2;
    float s2 = glm::length(glm::cross(sideA, sideB)) / s;

    sideB = p - v1;
    float s3 = glm::length(glm::cross(sideA, sideB)) / s;

    return glm::vec3(s1, s2, s3);
}

__host__ __device__ bool MollerTrumboreIntersect(
    const glm::vec3& rayOrigin,
    const glm::vec3& rayDir,
    const glm::vec3& v0,
    const glm::vec3& v1,
    const glm::vec3& v2,
    float& t,
    float& u,
    float& v)
{
    const float TRI_EPSILON = 0.0000001f;

    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 h = glm::cross(rayDir, edge2);
    float a = glm::dot(edge1, h);

    // Ray is parallel to triangle
    if (a > -TRI_EPSILON && a < TRI_EPSILON)
        return false;

    float f = 1.0f / a;
    glm::vec3 s = rayOrigin - v0;
    u = f * glm::dot(s, h);

    if (u < 0.0f || u > 1.0f)
        return false;

    glm::vec3 q = glm::cross(s, edge1);
    v = f * glm::dot(rayDir, q);

    if (v < 0.0f || u + v > 1.0f)
        return false;

    t = f * glm::dot(edge2, q);

    return t > EPSILON;
}


__host__ __device__ bool IntersectAABB(Ray& ray, glm::vec3 invDir, glm::vec3 boxMin, glm::vec3 boxMax, float t)
{

    glm::vec3 dir = ray.direction;
    glm::vec3 origin = ray.origin;

    float tx1 = (boxMin.x - origin.x) * invDir.x;
    float tx2 = (boxMax.x - origin.x) * invDir.x;
    float tmin = fminf(tx1, tx2), tmax = fmaxf(tx1, tx2);

    float ty1 = (boxMin.y - origin.y) * invDir.y;
    float ty2 = (boxMax.y - origin.y) * invDir.y;

    tmin = fmaxf(tmin, fminf(ty1, ty2)); tmax = fminf(tmax, fmaxf(ty1, ty2));

    float tz1 = (boxMin.z - origin.z) * invDir.z;
    float tz2 = (boxMax.z - origin.z) * invDir.z;

    tmin = fmaxf(tmin, fminf(tz1, tz2)); tmax = fminf(tmax, fmaxf(tz1, tz2));

    return tmax >= tmin && tmin < t && tmax > 0.f;

}

__host__ __device__ float IntersectAABB_GetTmin(Ray& ray, glm::vec3 invDir, glm::vec3 boxMin,
    glm::vec3 boxMax, float t_max)
{
    glm::vec3 dir = ray.direction;
    glm::vec3 origin = ray.origin;

    float tx1 = (boxMin.x - origin.x) * invDir.x;
    float tx2 = (boxMax.x - origin.x) * invDir.x;
    float tmin = fminf(tx1, tx2), tmax = fmaxf(tx1, tx2);

    float ty1 = (boxMin.y - origin.y) * invDir.y;
    float ty2 = (boxMax.y - origin.y) * invDir.y;
    tmin = fmaxf(tmin, fminf(ty1, ty2));
    tmax = fminf(tmax, fmaxf(ty1, ty2));

    float tz1 = (boxMin.z - origin.z)  * invDir.z;
    float tz2 = (boxMax.z - origin.z)  * invDir.z;
    tmin = fmaxf(tmin, fminf(tz1, tz2));
    tmax = fminf(tmax, fmaxf(tz1, tz2));

    if (tmax >= tmin && tmin < t_max && tmax > 0.f) {
        return tmin;  // Return entry distance
    }
    return FLT_MAX;  // No hit
}


__host__ __device__ float IntersectBVH(
    Ray& ray,
    const BVHNode* nodes,
    const Triangle* triangles,
    const int* tri_indices,
    int nodeIdx, 
    float t_max,
    int triangles_count,
    glm::vec3 &intersectP,
    glm::vec3 &normal,
    glm::vec2 &uv,
    int &matId) 
{

    float closest_t = FLT_MAX;
    glm::vec3 invDir = 1.0f / ray.direction;


    int stack[32];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    int tri_idx_near = -1;
    glm::vec3 coords_tmp;

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        const BVHNode& node = nodes[node_idx];


        if (!IntersectAABB(ray, invDir, node.aabbMin, node.aabbMax, closest_t)) {
            continue;
        }
        
        if (node.isLeaf()) {
            for (int i = 0; i < node.primCount; i++) {
                int triIdx = tri_indices[node.firstPrim + i];
                const Triangle& tri = triangles[triIdx];
                glm::vec3 baryCoords;


                float t, u, v;
                bool hit = MollerTrumboreIntersect(
                    ray.origin, ray.direction,
                    tri.v1.m_pos, tri.v2.m_pos, tri.v3.m_pos,
                    t, u, v
                );
                if (hit && t > 0.0f && t < closest_t) {
                    closest_t = t;
                    tri_idx_near = triIdx;
                    coords_tmp = glm::vec3(u,v,t);
                }
            }
        }
        else {
            int leftChild = node.leftChild;
            int rightChild = leftChild + 1;

            float tminLeft = IntersectAABB_GetTmin(ray, invDir, nodes[leftChild].aabbMin,
                nodes[leftChild].aabbMax, closest_t);
            float tminRight = IntersectAABB_GetTmin(ray, invDir, nodes[rightChild].aabbMin,
                nodes[rightChild].aabbMax, closest_t);

            bool hitLeft = (tminLeft < closest_t);
            bool hitRight = (tminRight < closest_t);

            if (hitLeft && hitRight) {
                if (tminLeft < tminRight) {
                    stack[stack_ptr++] = rightChild;
                    stack[stack_ptr++] = leftChild;
                }
                else {
                    stack[stack_ptr++] = leftChild;
                    stack[stack_ptr++] = rightChild;
                }
            }
            else if (hitLeft) {
                stack[stack_ptr++] = leftChild;
            }
            else if (hitRight) {
                stack[stack_ptr++] = rightChild;
            }
        }
    }

    if (closest_t == FLT_MAX || tri_idx_near == -1) {
        return -1.0f;
    }

    intersectP = ray.origin + (ray.direction * closest_t);

    Triangle tri_near = triangles[tri_idx_near];
    Vertex v1 = tri_near.v1;
    Vertex v2 = tri_near.v2;
    Vertex v3 = tri_near.v3;

    matId = tri_near.materialId;

    float u = coords_tmp.x;
    float v = coords_tmp.y;
    float w = 1.0f - u - v;


    // Interpolate normals
    normal = glm::normalize(w * v1.m_normal + u * v2.m_normal + v * v3.m_normal);
    uv = w * v1.m_uv + u * v2.m_uv + v * v3.m_uv;


    return glm::length(ray.origin - intersectP);


}


// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections, 
    Vertex* verts,
    int vert_size,
    BVHNode* bvh_nodes,
    Triangle* triangles,
    int* tri_indices,
    int triangles_count,
    bool enableBVH)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int obj_hit = -1;
        int hit_geom_index = -1;
        bool outside = true;
        glm::vec2 uv;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        int matId = -1;
        int meshType = -1;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            } 
            else if (geom.type == MESH)
            {
                if (enableBVH) {
                    t = IntersectBVH(pathSegment.ray, bvh_nodes, triangles, tri_indices, 0, t_min, triangles_count, tmp_intersect, tmp_normal, uv, matId);
                }
                else {
                    obj_hit = i;
                }
                meshType = 1;
            }
    
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;

            }

            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.

        }


        if (!enableBVH) {
            for (int vertIdx = 0; vertIdx < vert_size; vertIdx += 3)
            {
                Vertex& v1 = verts[vertIdx];
                Vertex& v2 = verts[vertIdx + 1];
                Vertex& v3 = verts[vertIdx + 2];
                glm::vec3 baryCoords;

                float t, u, v;
                bool hit = MollerTrumboreIntersect(
                    pathSegment.ray.origin, pathSegment.ray.direction,
                    v1.m_pos, v2.m_pos, v3.m_pos,
                    t, u, v
                );
 
                if (hit && t > 0.0f && t_min > t)
                {
                    t_min = t;
                    hit_geom_index = obj_hit;
                    intersect_point = pathSegment.ray.origin + (pathSegment.ray.direction * t_min);
                    
                    normal = glm::normalize(u * v1.m_normal + v * v2.m_normal + t * v3.m_normal); // Interpolate normals
                    uv = u * v1.m_uv + v * v2.m_uv + t * v3.m_uv; // Interpolate uv
                }
            }

        }
        
        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].surfaceUVCoord = uv;
            intersections[path_index].uv = uv;


            if (meshType == 1 && matId >= 0) {
                intersections[path_index].materialId = matId; // Triangle-level
            }
            else {
                intersections[path_index].materialId = geoms[hit_geom_index].materialid; // Geom-level
            }
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

__device__ glm::vec3 sampleEnvMapBilinear(
    const glm::vec3& dir,
    const Texture& envTex,
    glm::vec4* texels)
{
    glm::vec3 d = glm::normalize(dir);
    float theta = atan2f(d.z, d.x);
    float phi = acosf(glm::clamp(d.y, -1.0f, 1.0f));
    float u = (theta + M_PI) / (2.0f * M_PI);
    float v = phi / M_PI;

    // map to continuous coordinates
    float x = u * (envTex.width - 1);
    float y = v * (envTex.height - 1);

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float width = envTex.width;
	float height = envTex.height;

    x0 = glm::clamp((float)x0, 0.f, width - 1);
    y0 = glm::clamp((float)y0, 0.f, height - 1);
    x1 = glm::clamp((float)x1, 0.f, width - 1);
    y1 = glm::clamp((float)y1, 0.f, height - 1);

    float fx = x - floorf(x);
    float fy = y - floorf(y);

	float startIdx = envTex.startPixelTex;
    int w = envTex.width;

    glm::vec3 c00 = glm::vec3(texels[int(startIdx + y0 * w + x0)]);
    glm::vec3 c10 = glm::vec3(texels[int(startIdx + y0 * w + x1)]);
    glm::vec3 c01 = glm::vec3(texels[int(startIdx + y1 * w + x0)]);
    glm::vec3 c11 = glm::vec3(texels[int(startIdx + y1 * w + x1)]);

    glm::vec3 c0 = c00 * (1.0f - fx) + c10 * fx;
    glm::vec3 c1 = c01 * (1.0f - fx) + c11 * fx;
    glm::vec3 c = c0 * (1.0f - fy) + c1 * fy;

    return c;
}


__global__ void shadeDiffuseMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    bool enableRR,
    Texture* textures,
    glm::vec4* texels,
    Texture envTex,
    bool envtTrue
#if DENOISE
    , glm::vec3* dev_albedoImg,
    glm::vec3* dev_normalsImg
#endif
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    ShadeableIntersection intersection = shadeableIntersections[idx];

    // No intersection: make black and terminate
    if (intersection.t <= 0.0f) {
        if (envtTrue) {
            glm::vec3 envColor = sampleEnvMapBilinear(pathSegments[idx].ray.direction, envTex, texels);
            pathSegments[idx].color *= envColor;

            pathSegments[idx].remainingBounces = 0;
            return;
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
            return;
        }

        
    }

    // If path already dead, nothing to do
    if (pathSegments[idx].remainingBounces <= 0) {
        return;
    }

    // RNG: seed with iter and the path index and remainingBounces for variance
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
    thrust::uniform_real_distribution<float> u01(0, 1);

    Material material = materials[intersection.materialId];
    glm::vec3 materialColor = material.color;
    glm::vec3 normal = intersection.surfaceNormal;

    // Compute intersection position (world)
    glm::vec3 intersectPos = pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t;

    // If the hit material is emissive, accumulate emission into throughput and terminate path.
    if (material.emittance > 0.0f) {
        // Multiply throughput by emitted radiance (emissive color * emittance)
        pathSegments[idx].color *= (materialColor * material.emittance);

        pathSegments[idx].remainingBounces = 0;
#if DENOISE
        dev_albedoImg[pathSegments[idx].pixelIndex] = materialColor;
        dev_normalsImg[pathSegments[idx].pixelIndex] = normal;
#endif
        return;
    }

    // If textured, fetch texel color and override albedo (materialColor). Do not terminate path here.
    int texID = material.diffuseTextureID;
    if (texID >= 0) {
        Texture tex = textures[texID];

        // clamp uv to [0,1]
        float u = glm::clamp(intersection.uv.x, 0.0f, 1.0f);
        float v = 1.0f - glm::clamp(intersection.uv.y, 0.0f, 1.0f);

        // map uv to texel coords (nearest sampling). You may want bilinear later.
        int iu = static_cast<int>(u * (tex.width - 1));
        int iv = static_cast<int>(v * (tex.height - 1));

        iu = glm::clamp((float)iu, 0.f, tex.width - 1);
        iv = glm::clamp((float)iv, 0.f, tex.height - 1);

        int idxTex = tex.startPixelTex + iv * tex.width + iu;
        glm::vec4 texel = texels[idxTex];
        glm::vec3 texColor = glm::vec3(texel.x, texel.y, texel.z);

        materialColor = texColor;

    }

    if (material.hasReflective || material.hasRefractive) {
        // Perfect specular - no BRDF evaluation, throughput stays same
        pathSegments[idx].color *= material.color;

        scatterRay(pathSegments[idx], intersectPos, normal, material, rng);
        pathSegments[idx].remainingBounces--;
        return;
    }

    // Scatter ray: produce new direction in hemisphere about the normal
    scatterRay(pathSegments[idx], intersectPos, normal, material, rng);

    // Evaluate cosine between new direction and surface normal
    glm::vec3 wi = pathSegments[idx].ray.direction;
    float cosTheta = glm::dot(normal, wi);

    // If sampled direction goes below hemisphere (shouldn't with a correct hemisphere sampler)
    if (cosTheta <= 0.0f) {
        pathSegments[idx].remainingBounces = 0;
        return;
    }

    // Lambertian BRDF f_r = albedo / PI, PDF for cosine-weighted sampling = cosTheta / PI
    const float INV_PI = 1.0f / M_PI;
    glm::vec3 bsdf = materialColor * INV_PI;
    float pdf = cosTheta * INV_PI;

    // Update throughput: multiply by (f_r * cosTheta) / pdf
    // For cosine-weighted sampling, this reduces to materialColor (albedo), but keep explicit math.
    if (pdf > 0.0f) {
        glm::vec3 weight = (bsdf * cosTheta) / pdf; 
        pathSegments[idx].color *= weight;
    }
    else {
        pathSegments[idx].remainingBounces = 0;
        return;
    }

    pathSegments[idx].remainingBounces--;

#if DENOISE
    // Save albedo & normals (albedo should be the diffuse albedo, not the full throughput)
    // For denoising we want per-pixel albedo
    dev_albedoImg[pathSegments[idx].pixelIndex] = materialColor;
    dev_normalsImg[pathSegments[idx].pixelIndex] = normal;
#endif

    // Russian roulette 
    if (enableRR && pathSegments[idx].remainingBounces > 2) {
        // Choose survival probability based on maximum component of throughput (conservative)
        float maxComp = glm::max(pathSegments[idx].color.r, glm::max(pathSegments[idx].color.g, pathSegments[idx].color.b));
        // Keep p in a reasonable range
        float pSurvive = glm::clamp(maxComp, 0.05f, 0.95f);

        float xi = u01(rng);
        if (xi > pSurvive) {
            // terminate path
            pathSegments[idx].remainingBounces = 0;
            return;
        }
        else {
            // Compensate for probability of survival to keep estimator unbiased
            pathSegments[idx].color /= pSurvive;
        }
    }

}

#if DENOISE

__global__ void blendImages(int n, glm::vec3* noisy, glm::vec3* denoised, glm::vec3* output, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    output[i] = (1.0f - alpha) * noisy[i] + alpha * denoised[i];
}

void denoiseImage(glm::vec2 res, int pixelcount) {

    const int width = hst_scene->state.camera.resolution.x;
    const int height = hst_scene->state.camera.resolution.y;

    // Create an Open Image Denoise device
    oidn::DeviceRef device = oidn::newDevice(oidn::DeviceType::CPU);
    device.commit();

    std::vector<glm::vec3> h_image(pixelcount);
    std::vector<glm::vec3> h_normals(pixelcount);
    std::vector<glm::vec3> h_albedo(pixelcount);
    std::vector<glm::vec3> h_denoise(pixelcount);

    cudaMemcpy(h_image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_normals.data(), dev_normalsImg, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_albedo.data(), dev_albedoImg, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    oidn::FilterRef filter = device.newFilter("RT");  
    filter.setImage("color", h_image.data(), oidn::Format::Float3, width, height);
    filter.setImage("normal", h_normals.data(), oidn::Format::Float3, width, height);
    filter.setImage("albedo", h_albedo.data(), oidn::Format::Float3, width, height);
    filter.setImage("output", h_denoise.data(), oidn::Format::Float3, width, height);

    filter.set("hdr", true); 

    filter.commit();

    filter.execute(); 

    // Check for errors
    const char* errorMessage;
    if (device.getError(errorMessage) != oidn::Error::None) {
        std::cerr << "Error! " << errorMessage << std::endl;
    }

    cudaMemcpy(dev_denoiseImg, h_denoise.data(), pixelcount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
}

#endif

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}


struct compareMatId
{
    __host__ __device__
    bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const {
        return a.materialId < b.materialId;
    }
};

struct usefulPath
{
    __host__ __device__
    bool operator()(const PathSegment& seg) const {
        return seg.remainingBounces > 0;
    }
};


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter, bool materialSort, bool russianRoulette, bool enableBVH, bool antiAlias, bool dof, Texture envMap, bool envtTrue)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    /*
    // Track ray counts per bounce
    static std::vector<int> rayCountsPerBounce;
    if (iter == 1) { // Reset on first iteration
        rayCountsPerBounce.clear();
        rayCountsPerBounce.reserve(traceDepth + 1);
    }
    */

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    bool enableRR = false;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths, antiAlias, dof);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    /*
    // Record initial ray count
    if (iter == 1) {
        rayCountsPerBounce.push_back(num_paths);
    }
    */

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections,
            dev_vertices,
            (int)hst_scene->vertices.size(),
            dev_nodes,
            dev_triangles,
            dev_tri_indices,
            hst_scene->triangles.size(),
            enableBVH
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.


        // sort by material before performing shading and sampling
        // you can use thrust for this

        if (materialSort) {

            // shuffle path segments to be contiguous in memory and sort by material
            thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compareMatId());
        }

        if (russianRoulette) {
            // Russian Roulette Termination logic
            if (depth > 4) enableRR = true;
        }
        
        // shading directly in one big kernel
        shadeDiffuseMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            enableRR,
            dev_textures,
            dev_texels,
            envMap,
            envtTrue
#if DENOISE
            ,dev_albedoImg,
            dev_normalsImg
#endif
		);

   

        //TODO: Stream compact away all of the terminated paths.
        // You may use either your implementation or `thrust::remove_if` or its
        // cousins.

#if STREAM_COMPACT
        PathSegment* dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, usefulPath());
        num_paths = dev_path_end - dev_paths;

        /*
        // Record ray count after compaction
        if (iter == 1) {
            rayCountsPerBounce.push_back(num_paths);

            // Print to console
            printf("Depth %d: %d active rays (%.1f%%)\n",
                depth, num_paths, (num_paths * 100.0f) / pixelcount);
        }
        */
#endif
        
        if (num_paths == 0 || depth >= traceDepth) iterationComplete = true; // TODO: should be based off stream compaction results.
        

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

#if DENOISE
    int iterationCount = hst_scene->state.iterations;

    if (iter % DENOISE_ITERATION == 0) {
        denoiseImage(cam.resolution, pixelcount);
        // Copy denoised result to display buffer

        float alpha = 0.0f;
        if (iter > 16) {
            alpha = float(iter - 16) / float(iterationCount - 16);
            if (alpha > 1.0f) alpha = 1.0f;
        }

        // Blend noisy + denoised into dev_image 
        int blockSize = 128;
        int numBlocks = (pixelcount + blockSize - 1) / blockSize;
        blendImages << <numBlocks, blockSize >> > (pixelcount, dev_image, dev_denoiseImg, dev_image, alpha);
        cudaDeviceSynchronize();
    }
    else {
        // Copy raw accumulated image to display buffer
        cudaMemcpy(dev_displayImg, dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    }

    // Send display buffer to PBO
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_displayImg);
#else
    ///////////////////////////////////////////////////////////////////////////

// Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
#endif

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
