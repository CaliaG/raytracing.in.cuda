#include <iostream>
#include <fstream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "src/math/vector3D.h"
#include "src/utilities/ray.h"
#include "src/utilities/scene_builder.h"
#include "src/geometries/sphere.h"
#include "src/geometries/object_list.h"
#include "src/cameras/camera.h"
#include "src/materials/material.h"

#define STB_IMAGE_IMPLEMENTATION
#include "libs/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "libs/stb/stb_image_write.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// ------------------------------> SHOT
// -- EQUIVALENTE DI SHOT
// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vector3D color(const ray& r, object **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vector3D cur_attenuation = vector3D(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vector3D attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vector3D(0.0,0.0,0.0);
            }
        }
        else {
            vector3D unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vector3D c = (1.0f-t)*vector3D(1.0, 1.0, 1.0) + t*vector3D(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vector3D(0.0,0.0,0.0); // exceeded recursion
}

__device__ vector3D shot(const ray& r, object **world, int depth, curandState *local_rand_state) {
    ray cur_ray = r;
    vector3D cur_attenuation = vector3D(1.0,1.0,1.0);
    vector3D cur_emitted = vector3D(0.0,0.0,0.0);
    for(int i = 0; i < 8; i++) {
      hit_record rec;
      if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) { // equivalente di trace_ray
          ray scattered;
          vector3D attenuation;
          vector3D emitted = rec.mat_ptr->emitted(cur_ray, rec, rec.u, rec.v, rec.p);
          if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
              cur_emitted *= emitted;
              cur_attenuation *= attenuation;
              cur_ray = scattered;
              //return emitted + attenuation * shot(scattered, world, depth + 1, local_rand_state);
          }
          else {
              return emitted;
          }
      }
      else {
          vector3D unit_direction = unit_vector(cur_ray.direction());
          float t = 0.5f*(unit_direction.y() + 1.0f);
          vector3D c = (1.0f-t)*vector3D(1.0, 1.0, 1.0) + t*vector3D(0.5, 0.7, 1.0);
          return cur_emitted + cur_attenuation * c;
          //return vector3D(0.0,0.0,0.0);
      }
    }
    return vector3D(0.0,0.0,0.0); // exceeded recursion
}
// <------------------------------ SHOT

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

// -- EQUIVALENTE DI PARALLEL_RENDER
__global__ void render(vector3D *fb, int max_x, int max_y, int ns, camera **cam, object **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];

    vector3D col(0,0,0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v);
        col += shot(r, world, 0, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

__global__ void render_bkp(vector3D *fb, int max_x, int max_y, int ns, camera **cam, object **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];

    vector3D col(0,0,0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

__global__ void destroy(object **d_list, object **d_world, camera **d_camera, int obj_cnt) {
    /*for(int i = 0; i < obj_cnt; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }*/
    for (int i = 0; i < obj_cnt; i++) {
        delete *(d_list + i);
    }
    delete *d_world;
    delete *d_camera;
}

void get_device_props() {
    int nDevices;
  
    cudaGetDeviceCount(&nDevices);
    std::cerr << "+-----------------------------------------------------------------------------+" << std::endl;
    std::cerr << "                            CUDA DEVICE PROPERTIES" << std::endl;
    std::cerr << "+-----------------------------------------------------------------------------+" << std::endl;
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cerr << " Device Number: " << i << std::endl;
        std::cerr << "+=============================================================================+" << std::endl;
        std::cerr << " Device name: " << prop.name << std::endl;
        std::cerr << " Memory Clock Rate (KHz): "
                    << prop.memoryClockRate << std::endl;
        std::cerr << " Memory Bus Width (bits): "
                    << prop.memoryBusWidth << std::endl;
        std::cerr << " \tPeak Memory Bandwidth (GB/s): "
                    << 2.0f * prop.memoryClockRate *
                        (prop.memoryBusWidth / 8) / 1.0e6
                    << std::endl;
    }
    std::cerr << "+-----------------------------------------------------------------------------+" << std::endl;
}

void save_to_ppm(vector3D *fb, int nx, int ny) {
    std::ofstream ofs;
    ofs.open("./out.ppm", std::ios::out | std::ios::binary);
    ofs << "P3\n" << nx << " " << ny << "\n255\n";
        for (int j = ny-1; j >= 0; j--) {
            for (int i = 0; i < nx; i++) {
                size_t pixel_index = j*nx + i;
                int ir = int(255.99*fb[pixel_index].x()); // R
                int ig = int(255.99*fb[pixel_index].y()); // G
                int ib = int(255.99*fb[pixel_index].z()); // B
                ofs << ir << " " << ig << " " << ib << "\n";
            }
        }
    ofs.close();
}

int main() {
    int nx = 1200;
    int ny = 800;
    int ns = 32;
    int tx = 16;
    int ty = 16;

    get_device_props();

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vector3D);

    // allocate FB
    vector3D *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of objects & the camera
    object **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(object *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    // --> SPHERE SCENE
    //int num_objects = 22*22+1+3;      // senza instance
    ////int num_objects = 1;            // con instance
    //object **d_list;
    //checkCudaErrors(cudaMalloc((void **)&d_list, num_objects*sizeof(object *)));
    
    //build_random_scene<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
    //checkCudaErrors(cudaGetLastError());
    //checkCudaErrors(cudaDeviceSynchronize());
    // <-- SPHERE SCENE

    // --> BVH - SPHERE SCENE
    int num_objects = 22*22+1+3 + 1;
    object **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_objects*sizeof(object *)));
    
    build_bvh<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // <-- BVH - SPHERE SCENE

    // --> SIMPLE LIGHT
    /*ns = 500;
    int num_objects = 1;
    object **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_objects*sizeof(object *)));

    build_simple_light<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());*/
    // <-- SIMPLE LIGHT

    // --> CORNEL BOX
    /*ns = 500;
    int num_objects = 6*3 + 2;
    object **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_objects*sizeof(object *)));

    build_cornel_box<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());*/
    // <-- CORNEL BOX
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    std::cerr << "build scene took: " << time / 1000 << " seconds\n";

    cudaEventRecord(start, 0);

    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    std::cerr << "render init took: " << time / 1000 << " seconds\n";

    cudaEventRecord(start, 0);

    render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Output FB as Image
    save_to_ppm(fb, nx, ny);

    // Output FB as JPG
    // --> jpg
    // uint8_t* imgBuff = (uint8_t*)std::malloc(nx * ny * 3 * sizeof(uint8_t));
    // for (int j = ny - 1; j >= 0; --j) {
    //     for (int i = 0; i < nx; ++i) {
    //         size_t index = utils::XY(i, j);
    //         // stbi generates a Y flipped image
    //         size_t rev_index = utils::XY(i, HEIGHT - j - 1);
    //         float r = frameBuffer_u[index].r();
    //         float g = frameBuffer_u[index].g();
    //         float b = frameBuffer_u[index].b();
    //         imgBuff[rev_index * 3 + 0] = int(255.999f * r) & 255;
    //         imgBuff[rev_index * 3 + 1] = int(255.999f * g) & 255;
    //         imgBuff[rev_index * 3 + 2] = int(255.999f * b) & 255;
    //     }
    // }

    // stbi_write_png("render.png", nx, ny, 3, imgBuff, nx * 3);
    // stbi_write_jpg("out.jpg", nx, ny, 3, imgBuff, 100);
    // std::free(imgBuff);
    // <-- jpg


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    std::cerr << "took: " << time / 1000 << " seconds\n";

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    destroy<<<1,1>>>(d_list, d_world, d_camera, num_objects);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}
