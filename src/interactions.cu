#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    /*
    if (m.hasReflective && m.hasRefractive) {

        // Ideal DIELECTRIC interaction

        glm::vec3 wo = glm::normalize(pathSegment.ray.direction);
        float n1 = 1.0f; // air
        float n2 = m.indexOfRefraction; // material
        n2 = 1.1f;
        glm::vec3 n = normal;
        float cosTheta1 = glm::dot(-wo, n);
        if (cosTheta1 < 0) { // inside the surface
            n = -normal;
            cosTheta1 = glm::dot(-wo, n);
            float temp = n1;
            n1 = n2;
            n2 = temp;
        }
        float nRatio = n1 / n2;
        float sin2Theta2 = nRatio * nRatio * (1 - cosTheta1 * cosTheta1);
        if (sin2Theta2 > 1) { // total internal reflection
            glm::vec3 wi = glm::reflect(wo, n);
            pathSegment.ray.origin = intersect + 0.001f * n;
            pathSegment.ray.direction = glm::normalize(wi);
            pathSegment.remainingBounces--;
            return;
        }
        float cosTheta2 = sqrt(1 - sin2Theta2);
        float R0 = (n1 - n2) * (n1 - n2) / ((n1 + n2) * (n1 + n2));
        float R = R0 + (1 - R0) * pow((1 - cosTheta1), 5);
        thrust::uniform_real_distribution<float> u01(0, 1);
        if (u01(rng) < R) { // reflect
            glm::vec3 wi = glm::reflect(wo, n);
            pathSegment.ray.origin = intersect + 0.001f * n;
            pathSegment.ray.direction = glm::normalize(wi);
            pathSegment.remainingBounces--;
            return;
        }
        else { // refract
            glm::vec3 wi; // = nRatio * wo + (nRatio * cosTheta1 - cosTheta2) * n;
			wi = glm::refract(wo, n, nRatio);
            pathSegment.ray.origin = intersect - 0.001f * n;
            pathSegment.ray.direction = glm::normalize(wi);
            pathSegment.remainingBounces--;
            return;
        }


    }
    */
    if (m.hasReflective) {

        // Perfectly SPECULAR interaction

        glm::vec3 wo = glm::normalize(pathSegment.ray.direction);
        glm::vec3 wi = glm::reflect(wo, normal);
        pathSegment.ray.origin = intersect + 0.001f * normal;
        pathSegment.ray.direction = glm::normalize(wi);
        pathSegment.remainingBounces--;
        //return;
	}
    else {
        // Perfectly DIFFUSE interaction
        glm::vec3 wi = calculateRandomDirectionInHemisphere(normal, rng);

        pathSegment.ray.origin = intersect + 0.001f * normal;
        pathSegment.ray.direction = glm::normalize(wi);

        return;
    }
    
}
