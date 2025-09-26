#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    /*
    if (!outside)
    {
        normal = -normal;
    }
    */
    return glm::length(r.origin - intersectionPoint);
}



__host__ __device__ glm::vec3 barycentric(glm::vec3 p, glm::vec3 t1, glm::vec3 t2, glm::vec3 t3) {
    glm::vec3 edge1 = t2 - t1;
    glm::vec3 edge2 = t3 - t2;
    float S = length(cross(edge1, edge2));

    edge1 = p - t2;
    edge2 = p - t3;
    float S1 = length(cross(edge1, edge2));

    edge1 = p - t1;
    edge2 = p - t3;
    float S2 = length(cross(edge1, edge2));

    edge1 = p - t1;
    edge2 = p - t2;
    float S3 = length(cross(edge1, edge2));

    return glm::vec3(S1 / S, S2 / S, S3 / S);
}



__host__ __device__ float meshIntersectionTest(
    Geom mesh,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    glm::vec2& uv,
    Vertex* verts,
    int vert_size,
    float t_min)
{

    float t = FLT_MAX;
    ;

    for (int vertIdx = 0; vertIdx < vert_size; vertIdx += 3)
    {
        Vertex& v1 = verts[vertIdx];
        Vertex& v2 = verts[vertIdx + 1];
        Vertex& v3 = verts[vertIdx + 2];
        glm::vec3 baryCoords;
        bool hit = glm::intersectRayTriangle(r.origin, r.direction, v1.m_pos, v2.m_pos, v3.m_pos, baryCoords);
        t = baryCoords.z;

        if (t > 0.0f && t_min > t && hit)
        {
            t_min = t;
            //mat_index = v1.materialid;
            glm::vec3 p = r.origin + (r.direction * t);
            intersectionPoint = p;
            glm::vec3 bary = barycentric(p, v1.m_pos, v2.m_pos, v3.m_pos); // Calculate barycentric coordinates
            normal = glm::normalize(bary.x * v1.m_normal + bary.y * v2.m_normal + bary.z * v3.m_normal); // Interpolate normals
            uv = bary.x * v1.m_uv + bary.y * v2.m_uv + bary.z * v3.m_uv; // Interpolate uv
            //tmp_tangent = bary.x * v1.tangent + bary.y * v2.tangent + bary.z * v3.tangent;

        }

    }
    return t_min;


}

