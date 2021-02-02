#pragma once

#include "object.h"
#include "../utilities/bvh.h"

class object_list: public object  {
public:
    __device__ object_list() {}
    __device__ object_list(object **l, bvh_node* bvh_node, uint32_t size) {list = l; bvh = bvh_node; list_size = size; }
    __device__ ~object_list() noexcept override;
    
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    __device__ virtual bool hit_shadow(const ray& r, float t_min, float t_max) const;
    __device__ virtual bool bounding_box(aabb& box) const;

    __device__ constexpr object* get_object(uint32_t id);

private:
    object** list;
    bvh_node* bvh;
    uint32_t list_size;
};

__device__ object_list::~object_list() noexcept {
    for (uint32_t i = 0; i < list_size; ++i) {
        delete* (list + i);
    }
}

__device__ constexpr object* object_list::get_object(uint32_t id) {
    for (uint32_t i = 0; i < list_size; ++i) {
        if (list[i]->get_id() == id) {
            return list[i];
        }
    }
    return nullptr;
}

__device__ bool object_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    if (_bvh) {
        return(_bvh->dfs(r, tmin, tmax, hrec));
    } else {
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

__device__ bool object_list::bounding_box(aabb& box) const {
    if (list_size < 1) return false;

    aabb tempbbox;
    if (!list[0]->bounding_box(tempbbox)) {
        return false;
    } else {
        box = tempbbox;
    }

    // we make the bounding box bigger and bigger with each object
    for (uint32_t i = 1; i < _size; ++i) {
        if (list[i]->bounding_box(tempbbox)) {
            box = aabb::surrounding_box(box, tempbbox);
        } else {
            return false;
        }
    }
    return false;
}
