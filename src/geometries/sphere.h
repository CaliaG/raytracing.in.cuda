#pragma once

#include "object.h"

class sphere: public object  {
public:
    __device__ sphere() {
        center = point3D(0.0f, 0.0f, 0.0f);
		    radius = 1.0f;
    }
    __device__ sphere(point3D cen, float r, material *m) : center(cen), radius(r), mat_ptr(m)  {};
    __device__ virtual ~sphere() noexcept override;

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ virtual bool hit_shadow(const ray& r, float t_min, float t_max) const;
    __device__ virtual bool bounding_box(aabb& box) const;

//protected:
public:
    point3D center;
    float radius;
    material *mat_ptr;
};

__device__ sphere::~sphere() noexcept {
    if (mat_ptr) { delete mat_ptr; }
}

__device__ bool sphere::bounding_box(aabb& box) const {
	box = aabb(center - vector3D(radius, radius, radius), center + vector3D(radius, radius, radius));
	return true;
}

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vector3D oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}

__device__ bool sphere::hit_shadow(const ray& ray, float t_min, float t_max) const {
    vector3D oc = ray.origin() - center;
    float a = dot(ray.direction(), ray.direction());
    float b = dot(oc, ray.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            return true;
        }
    }
    return false;
}
