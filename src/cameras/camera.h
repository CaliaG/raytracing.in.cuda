#pragma once

#include <curand_kernel.h>
#include "../utilities/ray.h"

__device__ vector3D random_in_unit_disk(curandState *local_rand_state) {
    vector3D p;
    do {
        p = 2.0f*vector3D(curand_uniform(local_rand_state),curand_uniform(local_rand_state),0) - vector3D(1,1,0);
    } while (dot(p,p) >= 1.0f);
    return p;
}

/*
class camera_focus {

public:
    __device__ camera_focus(point3D lookfrom, point3D lookat, vector3D vup, float vfov, float aspect, float aperture, float focus_dist) { // vfov is top to bottom in degrees
        lens_radius = aperture / 2.0f;
        float theta = vfov*((float)M_PI)/180.0f;
        float half_height = tan(theta/2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        point3D tmp = origin - half_width*focus_dist*u - half_height*focus_dist*v - focus_dist*w;
        lower_left_corner = vector3D(tmp.x, tmp.y, tmp.z);
        horizontal = 2.0f*half_width*focus_dist*u;
        vertical = 2.0f*half_height*focus_dist*v;
        lower_left_corner = - half_width*u -half_height*v - w;
        horizontal = 2.0f*half_width*focus_dist*u;
        vertical = 2.0f*half_height*focus_dist*v;
    }

    __device__ ray get_ray(float s, float t, curandState *local_rand_state) {
        vector3D rd = lens_radius*random_in_unit_disk(local_rand_state);
        vector3D offset = u * rd.x() + v * rd.y();
        point3D tmp = lower_left_corner + s*horizontal + t*vertical - origin - offset;
        return ray(origin + offset, vector3D(tmp.x, tmp.y, tmp.z));
        //return ray(origin, lower_left_corner + s*horizontal + t*vertical);
    }

    point3D origin;
    vector3D lower_left_corner;
    vector3D horizontal;
    vector3D vertical;
    vector3D u, v, w;
    float lens_radius;
};
*/

class camera {

public:
    __device__ camera(point3D lookfrom, point3D lookat, vector3D vup, float vfov, float aspect) {
        float theta = vfov*((float)M_PI)/180.0f;
        float half_height = tan(theta/2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = - half_width*u -half_height*v - w;
        horizontal = 2.0f*half_width*u;
        vertical = 2.0f*half_height*v;
    }

    __device__ ray get_ray(float s, float t) {
        return ray(origin, lower_left_corner + s*horizontal + t*vertical);
    }

    point3D origin;
    vector3D lower_left_corner;
    vector3D horizontal, vertical;
    vector3D u, v, w;
};