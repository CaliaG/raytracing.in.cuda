#pragma once

#include "object.h"
#include "triangle.h"
#include "../math/math.h"
#include "../math/vector3D.h"
#include "../math/point3D.h"
#include "../math/aabb.h"

class rectangle : public object {
public:
    __device__ rectangle() {}
    __device__ rectangle(point3D v0, point3D v1, point3D v2, point3D v3) : tri_a(v0, v1, v3), tri_b(v1, v2, v3) {};
    
    __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record& rec) const;
    __device__ virtual bool hit_shadow(const ray& r, float t_min, float t_max) const;
    __device__ virtual bool bounding_box(aabb& box) const {
        aabb box_a, box_b;
        
        tri_a.bounding_box(box_a);
        tri_b.bounding_box(box_b);

        box = surrounding_box(box_a, box_b);

        return true;
    }
    
    triangle tri_a, tri_b;
};

__device__ bool rectangle::hit(const ray& r, float tmin, float tmax, hit_record &rec) const {
    if (tri_a.hit(r, tmin, tmax, rec))
        return true;
    else
        return tri_b.hit(r, tmin, tmax, rec);
}

__device__ bool rectangle::hit_shadow(const ray& r, float tmin, float tmax) const {
    if (tri_a.hit_shadow(r, tmin, tmax))
        return true;
    else
        return tri_b.hit_shadow(r, tmin, tmax);
}
