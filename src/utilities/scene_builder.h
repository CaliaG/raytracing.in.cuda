#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

#include "../cameras/camera.h"
#include "../geometries/object.h"
#include "../geometries/object_list.h"
#include "../geometries/instance.h"
#include "../geometries/sphere.h"
#include "../materials/material.h"
#include "../math/point3D.h"
#include "../math/vector3D.h"
#include "../utilities/texture.h"

#define RND (curand_uniform(&local_rand_state))

__device__ void random_scene(object **d_list, object **d_world, curandState *state) {
    curandState local_rand_state = *state;
    Texture *checker = new checkerTexture(new constantTexture(vector3D(0.5f, 0.7f, 0.8f)), new constantTexture(vector3D(0.9, 0.9, 0.9)));
    d_list[0] = new sphere(point3D(0, -1000.0, -1), 1000, new lambertianTexture(checker));
    // d_list[0] = new sphere(point3D(0, -1000.0, -1), 1000, new lambertian(vector3D(RND*RND, RND*RND, RND*RND)));           // green sphere
    // d_list[0] = new sphere(point3D(0, -1000.0, -1), 1000, new lambertian(vector3D(0.5, 0.5, 0.5)));                       // gray sphere
    int i = 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = RND;
            point3D center(a + 0.9 * RND, 0.2, b + 0.9 * RND);
            if (choose_mat < 0.8f) {
                // d_list[i++] = new movingSphere(center, center + point3D(0, 0.5*RND, 0), 0.0, 1.0, 0.2, new lambertianTexture(new constantTexture(vector3D(RND, RND, RND))));
                d_list[i++] = new sphere(center, 0.2, new lambertian(vector3D(RND*RND, RND*RND, RND*RND)));
                continue;
            }
            else if(choose_mat < 0.95f) {
                d_list[i++] = new sphere(center, 0.2,
                                       new metal(vector3D(0.5f*(1.0f + RND), 
                                                          0.5f*(1.0f + RND), 
                                                          0.5f*(1.0f + RND)), 
                                                          0.5f*RND));
            }
            else {
                d_list[i++] = new sphere(center, 0.2, new dielectric(RND*2));
            }
        }
    }
    d_list[i++] = new sphere(point3D( 0, 1, 0), 1.0, new dielectric(1.5));
    d_list[i++] = new sphere(point3D(-4, 1, 0), 1.0, new lambertian(vector3D(0.4, 0.2, 0.1)));
    d_list[i++] = new sphere(point3D( 4, 1, 0), 1.0, new metal(vector3D(0.7, 0.6, 0.5), 0.0));
    *state = local_rand_state;
    *d_world = new object_list(d_list, i);
}

__device__ void random_scene_instance(object **d_list, object **d_world, curandState *state) {
    curandState local_rand_state = *state;

    object* sphere_model = new sphere();

    // PIANO
    Texture *checker = new checkerTexture(new constantTexture(vector3D(0.5, 0.5, 0.5)), new constantTexture(vector3D(0.95f, 0.95f, 0.95f)));
    instance* sphere_ptr = new instance(sphere_model, new lambertianTexture(checker));
    sphere_ptr->scale(1000.0f, 1000.0f, 1000.0f);
    sphere_ptr->translate(0.0f, -1000.0f, 0.0f);
    d_list[0] = sphere_ptr;
    
    int i = 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = RND;
            point3D center(a + 0.9 * RND, 0.2, b + 0.9 * RND);
            if (choose_mat < 0.8f) {
                sphere_ptr = new instance(sphere_model, new lambertian(vector3D(RND*RND, RND*RND, RND*RND)));
                sphere_ptr->scale(0.2f, 0.2f, 0.2f);
                sphere_ptr->translate(center.x, center.y, center.z);
                d_list[i++] = sphere_ptr;
                continue;
            }
            else if(choose_mat < 0.95f) {
                sphere_ptr = new instance(sphere_model, new metal(vector3D(0.5f*(1.0f + RND), 
                                                                           0.5f*(1.0f + RND), 
                                                                           0.5f*(1.0f + RND)), 
                                                                           0.5f*RND));
                sphere_ptr->scale(0.2f, 0.2f, 0.2f);
                sphere_ptr->translate(center.x, center.y, center.z);
                d_list[i++] = sphere_ptr;
            }
            else {
                sphere_ptr = new instance(sphere_model, new dielectric(RND*2));
                sphere_ptr->scale(0.2f, 0.2f, 0.2f);
                sphere_ptr->translate(center.x, center.y, center.z);
                d_list[i++] = sphere_ptr;
            }
        }
    }
    sphere_ptr = new instance(sphere_model, new dielectric(1.5));
    sphere_ptr->translate(0.0f, 1.0f, 0.0f);
    d_list[i++] = sphere_ptr;
    sphere_ptr = new instance(sphere_model, new lambertian(vector3D(0.4, 0.2, 0.1)));
    sphere_ptr->translate(-4.0f, 1.0f, 0.0f);
    d_list[i++] = sphere_ptr;
    sphere_ptr = new instance(sphere_model, new metal(vector3D(0.7, 0.6, 0.5), 0.0));
    sphere_ptr->translate(4.0f, 1.0f, 0.0f);
    d_list[i++] = sphere_ptr;
    
    *state = local_rand_state;
    *d_world = new object_list(d_list, i);
}

__global__ void build_random_scene(object** obj_list, object** d_world, camera** d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    random_scene(obj_list, d_world, rand_state);

    point3D lookfrom(13,2,3);
    point3D lookat(0,0,0);
    //float dist_to_focus = 10.0;
    //float aperture = 0.1;
    float vfov = 30.0;
    vector3D up = vector3D(0,1,0);
    *d_camera = new camera(lookfrom, lookat, up, vfov, float(nx) / float(ny)/*, aperture, dist_to_focus*/);
    
    // // -- motion camera
    // vec3 lookfrom(0, 0, 10);
    // vec3 lookat(0, 0, 0);
    // float dist_to_focus = 10.0;
    // float aperture = 0.0;
    // float vfov = 60.0;
    // vec3 up = vec3(0,1,0);
    // *d_camera = new motionCamera(lookfrom, lookat, up, vfov, float(nx) / float(ny), aperture, dist_to_focus, 0.0f, 1.0f);
}
