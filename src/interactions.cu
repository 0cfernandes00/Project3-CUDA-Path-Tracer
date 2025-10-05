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
    const float RAY_EPSILON = 0.0001f;
    
    if (m.hasRefractive) {
        // Glass/Dielectric material
        glm::vec3 wo = pathSegment.ray.direction; // incoming ray (already normalized)

        float eta1 = 1.0f; // air
        float eta2 = m.indexOfRefraction; // glass (typically 1.5)

        glm::vec3 n = normal;
        float cosTheta = glm::dot(-wo, n); // Note the negative: angle between -wo and normal

        // Determine if entering or exiting the surface
        bool entering = cosTheta > 0.0f;
        if (!entering) {
            // Exiting: flip normal and swap IORs
            n = -n;
            cosTheta = -cosTheta;
            float temp = eta1;
            eta1 = eta2;
            eta2 = temp;
        }

        float etaRatio = eta1 / eta2;

        // Check for total internal reflection
        float sin2Theta2 = etaRatio * etaRatio * (1.0f - cosTheta * cosTheta);

        if (sin2Theta2 >= 1.0f) {
            // Total internal reflection
            pathSegment.ray.direction = glm::reflect(wo, n);
            pathSegment.ray.origin = intersect + RAY_EPSILON * n;
            return;
        }

        // Schlick's approximation for Fresnel
        float R0 = (eta1 - eta2) / (eta1 + eta2);
        R0 = R0 * R0;
        float R = R0 + (1.0f - R0) * powf(1.0f - cosTheta, 5.0f);

        // Randomly choose reflection or refraction based on Fresnel
        thrust::uniform_real_distribution<float> u01(0, 1);
        if (u01(rng) < R) {
            // Reflect
            pathSegment.ray.direction = glm::reflect(wo, n);
            pathSegment.ray.origin = intersect + RAY_EPSILON * n;
        }
        else {
            // Refract
            pathSegment.ray.direction = glm::refract(wo, n, etaRatio);
            // When refracting INTO the surface, move slightly inside (negative offset)
            pathSegment.ray.origin = intersect - RAY_EPSILON * n;
        }
        return;
    }
    if (m.hasReflective) {

        // Perfectly SPECULAR interaction

        glm::vec3 wo = glm::normalize(pathSegment.ray.direction);
        glm::vec3 wi = glm::reflect(wo, normal);
        pathSegment.ray.origin = intersect + 0.001f * normal;
        pathSegment.ray.direction = glm::normalize(wi);
        pathSegment.remainingBounces--;
        return;
	}
    else {
        // Perfectly DIFFUSE interaction
        glm::vec3 wi = calculateRandomDirectionInHemisphere(normal, rng);

        pathSegment.ray.origin = intersect + 0.001f * normal;
        pathSegment.ray.direction = glm::normalize(wi);

        return;
    }
    
}
