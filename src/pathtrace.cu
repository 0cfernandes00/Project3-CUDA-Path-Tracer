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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, bool antiAlias)
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

#if 0
        // lens effect Depth of Field      
        if (cam.lensRadius > 0) {
            // Sample point on lens

            // returns a vec3
                // TODO
            glm::vec2 sample(0,0);

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
#endif       
        

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

#if 0
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
#else
__host__ __device__ glm::vec3 barycentricCoords(glm::vec3 p, glm::vec3 a, glm::vec3 b, glm::vec3 c) {
    glm::vec3 v0 = b - a, v1 = c - a, v2 = p - a;
    float d00 = glm::dot(v0, v0);
    float d01 = glm::dot(v0, v1);
    float d11 = glm::dot(v1, v1);
    float d20 = glm::dot(v2, v0);
    float d21 = glm::dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;

    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;

    return glm::vec3(u, v, w);
}
#endif

__host__ __device__ bool IntersectAABB(Ray& ray, glm::vec3 boxMin, glm::vec3 boxMax, float t)
{

    glm::vec3 dir = ray.direction;
    glm::vec3 origin = ray.origin;

    float tx1 = (boxMin.x - origin.x) / dir.x;
    float tx2 = (boxMax.x - origin.x) / dir.x;
    float tmin = fminf(tx1, tx2), tmax = fmaxf(tx1, tx2);

    float ty1 = (boxMin.y - origin.y) / dir.y;
    float ty2 = (boxMax.y - origin.y) / dir.y;

    tmin = fmaxf(tmin, fminf(ty1, ty2)); tmax = fminf(tmax, fmaxf(ty1, ty2));

    float tz1 = (boxMin.z - origin.z) / dir.z;
    float tz2 = (boxMax.z - origin.z) / dir.z;

    tmin = fmaxf(tmin, fminf(tz1, tz2)); tmax = fminf(tmax, fmaxf(tz1, tz2));

    return tmax >= tmin && tmin < t && tmax > 0.f;

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

    int stack[128];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    int tri_idx_near = -1;

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        const BVHNode& node = nodes[node_idx];

        
        if (!IntersectAABB(ray, node.aabbMin, node.aabbMax, t_max)) {
            continue;
        }
        
        if (node.isLeaf()) {
            for (int i = 0; i < node.primCount; i++) {
                int triIdx = tri_indices[node.firstPrim + i];
                const Triangle& tri = triangles[triIdx];
                glm::vec3 baryCoords;

                bool hit = glm::intersectRayTriangle(ray.origin,ray.direction,tri.v1.m_pos, tri.v2.m_pos, tri.v3.m_pos, baryCoords);
                float out_t = baryCoords.z;
                if (hit && out_t > 0.0f && out_t < closest_t) {
                    closest_t = out_t;
                    tri_idx_near = triIdx;
                }
            }
        }
        else {
            stack[stack_ptr++] = node.leftChild + 1;
            stack[stack_ptr++] = node.leftChild;
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

    // Calculate barycentric coordinates
    glm::vec3 bary = barycentricCoords(intersectP, v1.m_pos, v2.m_pos, v3.m_pos); 

    // Interpolate normals
    normal = glm::normalize(bary.x * v1.m_normal + bary.y * v2.m_normal + bary.z * v3.m_normal); 
    uv = bary.x * v1.m_uv + bary.y * v2.m_uv + bary.z * v3.m_uv; // Interpolate uv


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

                /*if (geom.type == MESH && matId >= 0) {
                    intersections[path_index].materialId = matId;
                }*/
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
                bool hit = glm::intersectRayTriangle(pathSegment.ray.origin, pathSegment.ray.direction, v1.m_pos, v2.m_pos, v3.m_pos, baryCoords);
                t = baryCoords.z;

                if (hit && t > 0.0f && t_min > t)
                {
                    t_min = t;
                    hit_geom_index = obj_hit;
                    intersect_point = pathSegment.ray.origin + (pathSegment.ray.direction * t_min);

                    glm::vec3 bary = barycentricCoords(intersect_point, v1.m_pos, v2.m_pos, v3.m_pos); // Calculate barycentric coordinates
                    normal = glm::normalize(bary.x * v1.m_normal + bary.y * v2.m_normal + bary.z * v3.m_normal); // Interpolate normals
                    uv = bary.x * v1.m_uv + bary.y * v2.m_uv + bary.z * v3.m_uv; // Interpolate uv
                    //tmp_tangent = bary.x * v1.tangent + bary.y * v2.tangent + bary.z * v3.tangent;
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

__global__ void shadeDiffuseMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    bool enableRR,
    Texture* textures,
    glm::vec4* texels

#if DENOISE
    ,glm::vec3* dev_albedoImg,
    glm::vec3* dev_normalsImg 
#endif
)
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
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            if (pathSegments[idx].remainingBounces <= 0) {
                return;
			}

            int texID = materials[intersection.materialId].diffuseTextureID;

            /*
            if (texID >= 0) {
                Texture tex = textures[texID];

                int iu = glm::clamp(float(intersection.uv.x * tex.width), 0.f, tex.width - 1);
                int iv = glm::clamp(float(intersection.uv.y * tex.height), 0.f, tex.height - 1);

                int idxTex = tex.startPixelTex + iv * tex.width + iu;
                glm::vec4 texel = texels[idxTex];
                glm::vec3 texColor = glm::vec3(texel.x, texel.y, texel.z);

                pathSegments[idx].color = texColor;
                pathSegments[idx].remainingBounces = 0;
                //return;

                materialColor = texColor;
                //material.color = texColor;
                material.color = materialColor;
                // continue — do NOT return here


            }*/
            
#if 0
            if (material.diffuseTextureID != -1) {

                // read from texture

                Texture tmp = textures[material.diffuseTextureID];
                int startPixel = tmp.startPixelTex;
                glm::vec2 uv = intersection.uv;

                //int x = int(glm::fract(uv.x) * (float)tmp.width);
                //int y = int(glm::fract(1.0 - uv.y) * (float)tmp.height);

                int x = glm::min(float(glm::fract(uv.x) * tmp.width), tmp.width - 1.f);
                int y = glm::min(float(glm::fract(1.0f - uv.y) * tmp.height), tmp.height - 1.f);
                
                int texIdx = startPixel + y * tmp.width + x;

                //materialColor = glm::vec3(texels[texIdx]);
                glm::vec4 texel = texels[texIdx];
                materialColor = glm::vec3(texel.x, texel.y, texel.z);
                //materialColor = glm::vec3(uv.x, uv.y, 0.0f);

            }
            material.color = materialColor;
#endif           

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
				pathSegments[idx].remainingBounces = 0; // Terminate this path
            }

            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {

				// generate new ray direction with cosine-weighted hemisphere sampling
				glm::vec3 normal = intersection.surfaceNormal;

                if (texID >= 0) {
                    Texture tex = textures[texID];

                    int iu = glm::clamp(float(intersection.uv.x * tex.width), 0.f, tex.width - 1);
                    int iv = glm::clamp(float(intersection.uv.y * tex.height), 0.f, tex.height - 1);

                    int idxTex = tex.startPixelTex + iv * tex.width + iu;
                    glm::vec4 texel = texels[idxTex];
                    glm::vec3 texColor = glm::vec3(texel.x, texel.y, texel.z);

                    pathSegments[idx].color = texColor;
                    pathSegments[idx].remainingBounces = 0;
                    //return;

                    materialColor = texColor;
                    //material.color = texColor;
                    material.color = materialColor;
                    // continue — do NOT return here


                }
                else {
                    scatterRay(pathSegments[idx], pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t, normal, material, rng);
                }

                
                pathSegments[idx].color *= materialColor;
                pathSegments[idx].remainingBounces--;
                


                
#if DENOISE
               
                dev_albedoImg[pathSegments[idx].pixelIndex] = pathSegments[idx].color;
                dev_normalsImg[pathSegments[idx].pixelIndex] = intersection.surfaceNormal;

                if (enableRR) {
                    // find the path's maximum component output
                    float lMax = glm::max(pathSegments[idx].color.r, pathSegments[idx].color.g);
                    lMax = glm::max(lMax, pathSegments[idx].color.b);

                    // set the termination probability
                    thrust::uniform_real_distribution<float> u25(0,0.25);
                    float probStart = u25(rng);

                    float pTerm = (probStart > 1.0f - lMax) ? probStart : 1.0f - lMax;
                    float probSurvive = 1.0f - pTerm;

                    float xi = u01(rng);

                    // Roll the dice
                    if (xi < pTerm) {
                        // terminate ray
                        pathSegments[idx].remainingBounces = 0;
                        return;
                    } else {
                        pathSegments[idx].color /= probSurvive;
                    }
                }
#endif

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

#if DENOISE

__global__ void blendImages(int n, glm::vec3* noisy, glm::vec3* denoised, glm::vec3* output, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    output[i] = (1.0f - alpha) * noisy[i] + alpha * denoised[i];
}

__global__ void copyImage(glm::vec3* dest, glm::vec3* src, int pixelCount) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < pixelCount) {
        dest[index] = src[index];
    }
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
    //filter.set("cleanAux", true); 
    filter.commit();

    /*
    oidn::FilterRef albedoFilter = device.newFilter("RT"); 
    albedoFilter.setImage("albedo", dev_albedoImg, oidn::Format::Float3, width, height);
    albedoFilter.setImage("output", dev_albedoImg, oidn::Format::Float3, width, height);
    albedoFilter.commit();

    oidn::FilterRef normalFilter = device.newFilter("RT");
    normalFilter.setImage("normal", dev_normalsImg, oidn::Format::Float3, width, height);
    normalFilter.setImage("output", dev_normalsImg, oidn::Format::Float3, width, height);
    normalFilter.commit();
    */
    //albedoFilter.execute();
    //normalFilter.execute();


    filter.execute(); 

    // Check for errors
    const char* errorMessage;
    if (device.getError(errorMessage) != oidn::Error::None) {
        std::cerr << "Error! " << errorMessage << std::endl;
    }

    cudaMemcpy(dev_denoiseImg, h_denoise.data(), pixelcount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
}

__global__ void captureDenoiseData(
    int num_paths,
    ShadeableIntersection* intersections,
    PathSegment* paths,
    glm::vec3* normalsImg,
    glm::vec3* albedoImg,
    Material* materials,
    int iter) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_paths) {
    
        auto intersection = intersections[index];
        int pixelIdx = paths[index].pixelIndex;

        if (intersection.t > 0.0f) {
        
            Material mat = materials[intersection.materialId];
            normalsImg[pixelIdx] = intersection.surfaceNormal;
        
            glm::vec3 albedo = mat.color;
            if (mat.emittance > 0.0f) {
                albedo = mat.color * mat.emittance;
            }
            albedoImg[pixelIdx] = albedo;
        }
    
    }

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
void pathtrace(uchar4* pbo, int frame, int iter, bool materialSort, bool russianRoulette, bool enableBVH, bool antiAlias)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

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

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths, antiAlias);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

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

#if DENOISE
        if (depth == 1) {
            // Capture first bounce data for denoising
            /*
            captureDenoiseData << <numblocksPathSegmentTracing, blockSize1d >> > (
                num_paths,
                dev_intersections,
                dev_paths,
                dev_normalsImg,
                dev_albedoImg,
                dev_materials,
                iter
                );
            */
        }
#endif
        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        /*
        shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );*/

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
            dev_texels
#if DENOISE
            ,dev_albedoImg,
            dev_normalsImg
#endif
		);

   

        //TODO: Stream compact away all of the terminated paths.
        // You may use either your implementation or `thrust::remove_if` or its
        // cousins.
        PathSegment* dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, usefulPath());
        num_paths = dev_path_end - dev_paths;

        
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
        //cudaMemcpy(dev_displayImg, dev_denoiseImg, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
        float alpha = 0.0f;
        if (iter > 16) {
            alpha = float(iter - 16) / float(iterationCount - 16);
            if (alpha > 1.0f) alpha = 1.0f;
        }

        // Blend noisy + denoised into dev_image (the one you display)
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
