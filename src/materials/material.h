#pragma once

#include "../utilities/ray.h"
#include "../utilities/texture.h"
#include "../geometries/object.h"

struct hit_record;

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*pow((1.0f - cosine),5.0f);
}

__device__ bool refract(const vector3D& v, const vector3D& n, float ni_over_nt, vector3D& refracted) {
    vector3D uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > 0) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
        return false;
}

__device__ vector3D reflect(const vector3D& v, const vector3D& n) {
     return v - 2.0f*dot(v,n)*n;
}

#define RANDvector3D vector3D(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vector3D random_in_unit_sphere(curandState *local_rand_state) {
    vector3D p;
    do {
        p = 2.0f*RANDvector3D - vector3D(1,1,1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

class material  {
    public:
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3D& attenuation, ray& scattered, curandState *local_rand_state) const { return false; }
        __device__ virtual bool transmitted(const ray& r_in, const hit_record& rec, vector3D& attenuation, ray& scattered, curandState *local_rand_state) const { return false; }
        __device__ virtual vector3D emitted(const ray& r_in, const hit_record& rec, float u, float v, const point3D& p) const { return vector3D(0, 0, 0); }
	
    vector3D ka, kd, ks;
    float alpha;
};

class lambertian : public material {
    public:
        __device__ lambertian(const vector3D& a) : albedo(a) {}

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3D& attenuation, ray& scattered, curandState *local_rand_state) const  {
             vector3D target = /*rec.p + */rec.normal + random_in_unit_sphere(local_rand_state);
             scattered = ray(rec.p, target /*- rec.p*/);
             attenuation = albedo;
             return true;
        }

        vector3D albedo;
};

class lambertianTexture: public material {
public:
    __device__ lambertianTexture(Texture* a): albedo(a) {}

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3D& attenuation, ray& scattered, curandState *local_rand_state) const {
        point3D tmp = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
        vector3D target = vector3D(tmp.x, tmp.y, tmp.z);
        tmp = target-rec.p;
        scattered = ray(rec.p, vector3D(tmp.x, tmp.y, tmp.z));
        attenuation = albedo->value(0, 0, rec.p);
        return true;
    }

    Texture* albedo;
};

class metal : public material {
    public:
        __device__ metal(const vector3D& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3D& attenuation, ray& scattered, curandState *local_rand_state) const  {
            vector3D reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(local_rand_state));
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0.0f);
        }
        vector3D albedo;
        float fuzz;
};

class dielectric : public material {
public:
    __device__ dielectric(float ri) : ref_idx(ri) {}

    __device__ virtual bool scatter(const ray& r_in,
                         const hit_record& rec,
                         vector3D& attenuation,
                         ray& scattered,
                         curandState *local_rand_state) const  {
        vector3D outward_normal;
        vector3D reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = vector3D(1.0, 1.0, 1.0);
        vector3D refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction(), rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = sqrt(1.0f - ref_idx*ref_idx*(1-cosine*cosine));
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0f;
        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = ray(rec.p, reflected);
        else
            scattered = ray(rec.p, refracted);
        return true;
    }

    float ref_idx;
};

class diffuseLight : public material {
public:
    __device__ diffuseLight(Texture *a) : emit(a) {}

    __device__ virtual vector3D emitted(const ray& r_in, const hit_record& rec, float u, float v, const point3D& p) const {
        if (dot(rec.normal, r_in.direction()) < 0.0)
            return emit->value(u, v, p);
        else
            return vector3D(0, 0, 0);
    }

    Texture *emit;
};

#define RND (curand_uniform(&local_rand_state))

class specular : public material {
public:
    //specular(color ambient, color diffuse, color specular, float a) : ka(ambient), kd(diffuse), ks(specular), alpha(a) {};

    __device__ specular(curandState *state) {
        curandState local_rand_state = *state;
        ka = vector3D(RND, RND, RND);
        kd = vector3D(RND, RND, RND);
        ks = vector3D(RND, RND, RND);
        alpha = 20 + RND * 200;
        *state = local_rand_state;
    }

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3D& attenuation, ray& scattered, curandState *local_rand_state) const {
        return true;
    }

    curandState *state;
};
