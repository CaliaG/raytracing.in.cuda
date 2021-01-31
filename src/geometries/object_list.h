#pragma once

#include "object.h"

class object_list: public object  {
    public:
        __device__ object_list() {}
        __device__ object_list(object **l, int n) {list = l; list_size = n; }
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        __device__ virtual bool hit_shadow(const ray& r, float t_min, float t_max) const;
        __device__ virtual bool bounding_box(aabb& box) const;
        object **list;
        int list_size;
};

__device__ bool object_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;
        for (int i = 0; i < list_size; i++) {
            if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
}

__device__ bool object_list::hit_shadow(const ray& r, float t_min, float t_max) const {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;
        for (int i = 0; i < list_size; i++) {
            if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
            }
        }
        return hit_anything;
}

// TODO: correggere
 __device__ bool object_list::bounding_box(aabb& box) const {
     return true;
 }
