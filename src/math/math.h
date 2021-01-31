#pragma once

#include <stdlib.h>
// #include "color.h"
#include "vector3D.h"

extern int fail;

const float Deg2Rad = 3.1415926f / 180.0f;

__host__ __device__ inline float lerp(float a, float b, float t) {
	return (1.0f - t) * a + t * b;
}

__host__ __device__ inline vector3D lerp(vector3D &a, vector3D &b, float t) {
	return (1.0f - t) * a + t * b;
}

/*
__host__ __device__ inline color lerp(color &a, color &b, float t) {
	return (1.0f - t) * a + t * b;
}
*/

/*
__host__ __device__ float randZeroToOne() {
	return float(rand() / (RAND_MAX + 1.));
}

__host__ __device__ double randMToN(double M, double N) {
	  return M + (rand() / (RAND_MAX / (N - M)));
}

__host__ __device__ double randMinusOneToOne() {
	  return randMToN(-1.0, 1.0);
}
*/

/*
float max(const float a, const float b) {
	  return (a < b) ? b : a;
}
*/

// TODO: stesso codice di camera.h
/*
__host__ __device__ vector3D random_in_unit_sphere() {
    vector3D p;
    
    do {
        p = 2.0f*vector3D(randZeroToOne(), randZeroToOne(), randZeroToOne()) - vector3D(1, 1, 1);
    } while (magnitude(p) >= 1.0f);
    return p;
}
*/

__host__ __device__ inline float ffmin(float a, float b) { return a < b ? a : b; }
__host__ __device__ inline float ffmax(float a, float b) { return a > b ? a : b; }
