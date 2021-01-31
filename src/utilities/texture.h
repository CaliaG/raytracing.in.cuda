#pragma once

#include <cmath>
#include "../math/vector3D.h"
#include "../math/point3D.h"

class Texture {
public:
    __device__ virtual vector3D value(float u, float v, const point3D& p) const = 0;
};

class constantTexture: public Texture {
public: 
    __device__ constantTexture() {}
    __device__ constantTexture(vector3D c): color(c) {};

    __device__ virtual vector3D value(float u, float v, const point3D& p) const { return color; }

    vector3D color;
};

class checkerTexture: public Texture {
public:
    __device__ checkerTexture() {}
    __device__ checkerTexture(Texture* t0, Texture* t1): even(t0), odd(t1) {}

    __device__ virtual vector3D value(float u, float v, const point3D& p) const;

    Texture* odd;
    Texture* even;
};

__device__ vector3D checkerTexture::value(float u, float v, const point3D& p) const {
    float sines = sin(10 * p.x)*sin(10 * p.y)*sin(10 * p.z);
    if (sines < 0) {
        return odd->value(u, v, p);
    }
    else {
        return even->value(u, v, p);
    }
}

__device__ void get_sphere_uv(const point3D& p, float& u, float& v){
    float phi = atan2(p.z, p.x);
    float theta = asin(p.z);
    u = 1 - (phi + M_PI) / (2 * M_PI);
    v = (theta + M_PI/2) / M_PI;
}

class imageTexture: public Texture {
public:
    __device__ imageTexture() {}
    __device__ imageTexture(unsigned char* pixels, int imageWidth, int imageHeight): data(pixels), nx(imageWidth), ny(imageHeight) {}

    __device__ virtual vector3D value(float u, float v, const point3D& p) const;

    unsigned char* data;
    int nx, ny;
};

__device__ vector3D imageTexture::value(float u, float v, const point3D& p) const {
    int i = u * nx;
    int j = (1-v) * ny - 0.001f;
    if (i < 0) i = 0;
    if (j < 0) j = 0;
    if (i > nx - 1) i = nx - 1;
    if (j > ny - 1) j = ny - 1;
    
    float r = int(data[3*i + 3*nx*j  ]) / 255.0;
    float g = int(data[3*i + 3*nx*j+1]) / 255.0;
    float b = int(data[3*i + 3*nx*j+2]) / 255.0;
    return vector3D(r, g, b);
}