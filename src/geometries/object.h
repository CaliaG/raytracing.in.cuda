#pragma once

#include "../utilities/ray.h"
#include "../math/aabb.h"

class material;

struct hit_record
{
    float u;
    float v;
    float t;
    point3D p;
    vector3D normal;
    material *mat_ptr;
};

class object  {
public:
    __device__ object() {}

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
	  __device__ virtual bool hit_shadow(const ray& r, float t_min, float t_max) const = 0;
    __device__ virtual bool bounding_box(aabb& box) const = 0;
};
