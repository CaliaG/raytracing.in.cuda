#pragma once

#include "object.h"
#include "../math/math.h"
#include "../math/vector3D.h"
#include "../math/point3D.h"

class box : public object {
public:
    __device__ box() {
        pmin = point3D(-1.0, -1.0, -1.0);
        pmax = point3D(1.0, 1.0, 1.0);
    }

    __device__ box(point3D min, point3D max, material* m) {
        pmin = min;
        pmax = max;
        mat_ptr = m;
    }
    __device__ virtual ~box() noexcept override;

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ virtual bool hit_shadow(const ray& r, float t_min, float t_max) const;
    __device__ virtual bool bounding_box(aabb& box) const;

public:
    point3D pmin;
    point3D pmax;
    material *mat_ptr;
};

__device__ box::~box() noexcept {
    if (mat_ptr) { delete mat_ptr; }
}

__device__ bool box::hit(const ray& r, float tmin, float tmax, hit_record &rec) const {
    for (int i = 0; i < 3; i++) {
        float t0 = ffmin((pmin[i] - r.origin()[i]) / r.direction()[i],
                        (pmax[i] - r.origin()[i]) / r.direction()[i]);
        float t1 = ffmax((pmin[i] - r.origin()[i]) / r.direction()[i],
                         (pmax[i] - r.origin()[i]) / r.direction()[i]);
        tmin = ffmax(t0, tmin);
        tmax = ffmin(t1, tmax);
        if (tmax <= tmin)
            return false;
    }
    return true;
}

__device__ bool box::hit_shadow(const ray& r, float tmin, float tmax) const {
    for (int i = 0; i < 3; i++) {
        float t0 = ffmin((pmin[i] - r.origin()[i]) / r.direction()[i],
            (pmax[i] - r.origin()[i]) / r.direction()[i]);
        float t1 = ffmax((pmin[i] - r.origin()[i]) / r.direction()[i],
            (pmax[i] - r.origin()[i]) / r.direction()[i]);
        tmin = ffmax(t0, tmin);
        tmax = ffmin(t1, tmax);
        if (tmax <= tmin)
            return false;
    }
    return true;
}

__device__ bool box::bounding_box(aabb& box) const {
    box = aabb(pmin, pmax);
    return true;
}