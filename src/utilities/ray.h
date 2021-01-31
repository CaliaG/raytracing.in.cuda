#pragma once

#include "../math/vector3D.h"
#include "../math/point3D.h"

class ray {
public:
    point3D o;
    vector3D d;

    __device__ ray() {}
	  __device__ ray(const ray& ray) : o(ray.o), d(ray.d) {};
	  __device__ ray(const point3D& _origin, const vector3D& _direction) { o = _origin; d = _direction; }
	  __device__ point3D origin() const { return o; }
	  __device__ vector3D direction() const { return d; }
	  __device__ point3D point_at_parameter(float t) const { return o + t * d; }
};
