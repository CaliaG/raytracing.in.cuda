#pragma once

#include <thrust/sort.h>

#include "object.h"
#include "../math/aabb.h"
#include "../utilities/ray.h"

struct hit_record;

struct boxCompare {
    __device__ boxCompare(int m): mode(m) {}
    __device__ bool operator()(object* a, object* b) const {
        aabb box_left, box_right;
        object* ah = a;
        object* bh = b;
        
        if (!ah->bounding_box(box_left) || !bh->bounding_box(box_right)) {
            return false;
        }

        float val1, val2; 
        if (mode == 1) {
            val1 = box_left.pmin.x;
            val2 = box_right.pmin.x;
        } else if(mode == 2) {
            val1 = box_left.pmin.y;
            val2 = box_right.pmin.y;
        } else if(mode == 3) {
            val1 = box_left.pmin.z;
            val2 = box_right.pmin.z;
        }

        if (val1 - val2 < 0.0) {
            return false;
        } else {
            return true;
        }
    }
    int mode;
};

class bvhNode: public object {
public:
    __device__ bvhNode() {}
    __device__ bvhNode(object **d_list, int n, curandState *state);

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
	  __device__ virtual bool hit_shadow(const ray& r, float t_min, float t_max) const;
    __device__ virtual bool bounding_box(aabb& box) const;
    
    object* left;
    object* right;
    aabb box;
};

__device__ bvhNode::bvhNode(object **d_list, int n, curandState *state) {
    int axis = int(3 * curand_uniform(state));

    if (axis == 0){
        thrust::sort(d_list, d_list + n, boxCompare(1));
    } else if (axis == 1) {
        thrust::sort(d_list, d_list + n, boxCompare(2));
    } else {
        thrust::sort(d_list, d_list + n, boxCompare(3));
    }

    if (n == 1) {
        left = right = d_list[0];
    } else if (n == 2) {
        left  = d_list[0];
        right = d_list[1];
    } else {
        left  = new bvhNode(d_list, n/2, state);
        right = new bvhNode(d_list + n/2, n - n/2, state);
    }
    
    aabb box_left, box_right;
    if (!left->bounding_box(box_left) || !right->bounding_box(box_right)) {
        return;
        // std::cerr << "no bounding box in bvhNode constructor \n";
    }
    box = surrounding_box(box_left, box_right);
}

__device__ bool bvhNode::bounding_box(aabb& b) const {
    b = box;
    return true;
}

__device__ bool bvhNode::hit(const ray& r, float t_min, float t_max, hit_record& rec) const{
    if (box.hit(r, t_min, t_max)) {
        hit_record left_rec, right_rec;
        bool hit_left  = left->hit(r, t_min, t_max, left_rec);
        bool hit_right = right->hit(r, t_min, t_max, right_rec);
        if (hit_left && hit_right) {
            if (left_rec.t < right_rec.t) {
                rec = left_rec;
            } else {
                rec = right_rec;
            }
            return true;
        } else if (hit_left) {
            rec = left_rec;
            return true;
        } else if (hit_right) {
            rec = right_rec;
            return true;
        } else {
            return false;
        }
    }
    return false;
}

__device__ bool bvhNode::hit_shadow(const ray& r, float t_min, float t_max) const{
    if (box.hit(r, t_min, t_max)) {
        hit_record left_rec, right_rec;
        bool hit_left  = left->hit(r, t_min, t_max, left_rec);
        bool hit_right = right->hit(r, t_min, t_max, right_rec);
        if (hit_left && hit_right) {
            return true;
        } else if (hit_left) {
            return true;
        } else if (hit_right) {
            return true;
        } else {
            return false;
        }
    }
    return false;
}
