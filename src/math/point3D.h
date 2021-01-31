#pragma once

#include "vector3D.h"

class point3D {

public:
	  float x, y, z;

    __host__ __device__ point3D() {};
    __host__ __device__ point3D(float a, float b, float c) { x = a; y = b; z = c; }

    __host__ __device__ float& operator [](int i) { return ((&x)[i]); }
    __host__ __device__ const float& operator [](int i) const { return ((&x)[i]); }
};

__host__ __device__ inline vector3D operator -(const point3D& a, const point3D& b) {
	  return (vector3D(a.x - b.x, a.y - b.y, a.z - b.z));
}

__host__ __device__ inline point3D operator +(const point3D& a, const vector3D& b) {
	  return point3D(a.x + b.x(), a.y + b.y(), a.z + b.z());
}

__host__ __device__ inline point3D operator +(const vector3D& b, const point3D& a) {
	  return point3D(a.x + b.x(), a.y + b.y(), a.z + b.z());
}

__host__ __device__ inline point3D operator -(const point3D& a, const vector3D& b) {
	  return point3D(a.x - b.x(), a.y - b.y(), a.z - b.z());
}

__host__ __device__ inline point3D operator -(const vector3D& a, const point3D& b) {
	  return point3D(a.x() - b.x, a.y() - b.y, a.z() - b.z);
}
