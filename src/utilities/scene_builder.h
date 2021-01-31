#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

#include "../cameras/camera.h"
#include "../geometries/object.h"
#include "../geometries/object_list.h"
#include "../geometries/instance.h"
#include "../geometries/sphere.h"
#include "../geometries/rectangle.h"
#include "../materials/material.h"
#include "../math/point3D.h"
#include "../math/vector3D.h"
#include "../utilities/texture.h"

#define RND (curand_uniform(&local_rand_state))

// ---- SPHERE SCENE
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

    // piano
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

// ---- SIMPLE LIGHT
__device__ void simple_light(object **d_list, object **d_world, curandState *state) {
    object* sphere_model = new sphere();

    int i = 0;
    Texture *checker = new checkerTexture(new constantTexture(vector3D(0.2f, 0.3f, 0.1f)), new constantTexture(vector3D(0.9f, 0.9f, 0.9f)));
    instance* sphere_ptr = new instance(sphere_model, new lambertianTexture(checker));
    sphere_ptr->scale(1000.0f, 1000.0f, 1000.0f);
    sphere_ptr->translate(0.0f, -1000.0f, 0.0f);
    d_list[i++] = sphere_ptr;

    //sphere_ptr = new instance(sphere_model, new lambertianTexture(new constantTexture(color(0.4f, 0.2f, 0.1f))));
    sphere_ptr = new instance(sphere_model, new metal(vector3D(0.7f, 0.6f, 0.5f), 0.0f));
    //sphere_ptr = new instance(sphere_model, new dielectric(1.5f));
    sphere_ptr->scale(2.0f, 2.0f, 2.0f);
    sphere_ptr->translate(0, 2, 0);
    d_list[i++] = sphere_ptr;

    sphere_ptr = new instance(sphere_model, new diffuseLight(new constantTexture(vector3D(4.0f, 4.0f, 4.0f))));
    sphere_ptr->scale(2.0f, 2.0f, 2.0f);
    sphere_ptr->translate(0, 7, 0);
    d_list[i++] = sphere_ptr;

    point3D v0(4.0, 8.0, -4.0);
    point3D v1(4.0, 4.0, -4.0);
    point3D v2(4.0, 4.0,  4.0); 
    point3D v3(4.0, 8.0,  4.0);
    
    object* rectangle_model = new rectangle(v0, v1, v2, v3);
    instance* rectangle_ptr = new instance(rectangle_model, new diffuseLight(new constantTexture(vector3D(14.0f, 14.0f, 14.0f))));
    d_list[i++] = rectangle_ptr;

    *d_world = new object_list(d_list, i);
}

__global__ void build_simple_light(object** obj_list, object** d_world, camera** d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    simple_light(obj_list, d_world, rand_state);

    point3D lookfrom(-10, 2, 43);
    point3D lookat(0,0,0);
    float vfov = 20.0;
    vector3D up = vector3D(0,1,0);
    *d_camera = new camera(lookfrom, lookat, up, vfov, float(nx) / float(ny));
}

// ---- CORNEL BOX
__device__ void cornel_box(object **d_list, object **d_world, curandState *state) {
    material *red = new lambertianTexture(new constantTexture(vector3D(0.65, 0.05, 0.05)));
    material *white = new lambertianTexture(new constantTexture(vector3D(0.73, 0.73, 0.73)));
    material *green = new lambertianTexture(new constantTexture(vector3D(0.12, 0.45, 0.15)));
    material *light = new diffuseLight(new constantTexture(vector3D(2, 2, 2)));
    
    int i = 0;
    object* rectangle_model = new rectangle(point3D(343.0, 548.8, 227.0), point3D(343.0, 548.8, 332.0), point3D(213.0, 548.8, 332.0), point3D(213.0, 548.8, 227.0));
    instance* rectangle_ptr = new instance(rectangle_model, light);
    d_list[i++] = rectangle_ptr;

    // Back wall
    rectangle_model = new rectangle(point3D(549.6, 0.0, 559.2), point3D(0.0, 0.0, 559.2), point3D(0.0, 548.8, 559.2), point3D(556.0, 548.8, 559.2));
    rectangle_ptr = new instance(rectangle_model, white);
    d_list[i++] = rectangle_ptr;

    // Right wall
    rectangle_model = new rectangle(point3D(0.0, 0.0, 559.2), point3D(0.0, 0.0, 0.0), point3D(0.0, 548.8, 0.0), point3D(0.0, 548.8, 559.2));
    rectangle_ptr = new instance(rectangle_model, green);
    d_list[i++] = rectangle_ptr;

    // Left wall
    rectangle_model = new rectangle(point3D(552.8, 0.0, 0.0), point3D(549.6, 0.0, 559.2), point3D(556.0, 548.8, 559.2), point3D(556.0, 548.8, 0.0));
    rectangle_ptr = new instance(rectangle_model, red);
    d_list[i++] = rectangle_ptr;

    // Ceiling
    rectangle_model = new rectangle(point3D(556.0, 548.8, 0.0), point3D(556.0, 548.8, 559.2), point3D(0.0, 548.8, 559.2), point3D(0.0, 548.8, 0.0));
    rectangle_ptr = new instance(rectangle_model, white);
    d_list[i++] = rectangle_ptr;

    // Floor
    rectangle_model = new rectangle(point3D(552.8, 0.0, 0.0), point3D(0.0, 0.0, 0.0), point3D(0.0, 0.0, 559.2), point3D(549.6, 0.0, 559.2));
    rectangle_ptr = new instance(rectangle_model, white);
    d_list[i++] = rectangle_ptr;

    //sphere_ptr = new instance(sphere_model, new lambertian(new constant_texture(vector3D(0.4f, 0.2f, 0.1f))));
    //sphere_ptr = new instance(sphere_model, new metal(vector3D(0.7f, 0.6f, 0.5f), 0.0f));
    object* sphere_model = new sphere();
    instance* sphere_ptr = new instance(sphere_model, new dielectric(1.5f));
    //instance* sphere_ptr = new instance(sphere_model, light);
    sphere_ptr->scale(100.0f, 100.0f, 100.0f);
    sphere_ptr->translate(150, 150, 250);
    d_list[i++] = sphere_ptr;

    //sphere_ptr = new instance(sphere_model, new diffuse_light(new constant_texture(vector3D(4.0f, 4.0f, 4.0f))));
    sphere_ptr = new instance(sphere_model, new metal(vector3D(0.7f, 0.6f, 0.5f), 0.0f));
    sphere_ptr->scale(100.0f, 100.0f, 100.0f);
    sphere_ptr->translate(400, 100, 200);
    d_list[i++] = sphere_ptr;

    //object* model3d = new mesh("../models/bunny2.obj", "../models/");
    //instance* mesh_ptr = new instance(model3d, light);
    //mesh_ptr->scale(900.0f, 900.0f, 900.0f);
    //mesh_ptr->rotate_y(90.0f);
    //mesh_ptr->translate(250.0f, 150.0f, 500.0f);
    //d_list[i++] = mesh_ptr;
    
    *d_world = new object_list(d_list, i);
}

__global__ void build_cornel_box(object** obj_list, object** d_world, camera** d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    cornel_box(obj_list, d_world, rand_state);

    point3D lookfrom(278, 278, -800);
    point3D lookat(278, 278, 0);
    vector3D up(0, 1, 0);
    float vfov = 40.0;
    *d_camera = new camera(lookfrom, lookat, up, vfov, float(nx) / float(ny));
}
