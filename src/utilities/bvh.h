#pragma once

#include "../geometries/object.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>

#define STACK_SIZE 32

// class for bounding volume hierarchy
// a bvh node will check if a ray hits it or not
class bvh_node : public object {
public:
    __device__ bvh_node() {}
    __device__ bvh_node(object** hlist, int n, curandState* rstate, int level);

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ virtual bool hit_shadow(const ray& r, float t_min, float t_max) const;
    __device__ bool dfs(const ray& r, float tmin, float tmax, hit_record& hrec);

    __device__ virtual bool bounding_box(aabb& box) const;

    __device__ constexpr object* left() const { return _left; }
    __device__ constexpr object* right() const { return _right; }

    __device__ static void display_tree(bvh_node* root, int level);
    __device__ inline bool is_lowest_bvh() {
        // _left and _right should not be null
        return (_left->is_leaf() || _right->is_leaf());
    }

private:
    object* _left = nullptr;
    object* _right = nullptr;
    aabb _box;
};

struct box_compare {
    __device__ box_compare(int axis) : _axis(axis) {}

    __device__ bool operator()(object* ah, object* bh) const {
        aabb left_box, right_box;
        if (!ah->bounding_box(left_box)
            || !bh->bounding_box(right_box)) {
            //printf("Error: No bounding box in bvh_node constructor\n");
            return false;
        }

        float left_min, right_min;
        if (_axis == 0) {
            left_min = left_box.pmin.x;
            right_min = right_box.pmin.x;
        } else if (_axis == 1) {
            left_min = left_box.pmin.y;
            right_min = right_box.pmin.y;
        } else if (_axis == 2) {
            left_min = left_box.pmin.z;
            right_min = right_box.pmin.z;
        } else {
            //printf("Error: Unsupported comparison mode\n");
            return false;
        }

        if ((left_min - right_min) < 0.f) {
            return false;
        } else {
            return true;
        }
        return false;
    }
    int _axis = 0;
};

__device__ bvh_node::bvh_node(object** hlist, int n, curandState* rstate, int level) {
    // chose a random axis
    int axis = curand(rstate) % 3;
    if (axis == 0) {
        thrust::sort(hlist, hlist+n-1, box_compare(0));
    } else if (axis == 1) {
        thrust::sort(hlist, hlist + n - 1, box_compare(1));
    } else {
        thrust::sort(hlist, hlist + n - 1, box_compare(2));
    }

    // if one element, left and right are the same
    if (n == 1) {
        _left = _right = hlist[0];
    } else if (n == 2) {
        _left = hlist[0];
        _right = hlist[1];
    } else {
        _left = new bvh_node(hlist, n / 2, rstate, level + 1);
        _left->set_id((level + 1) * 10);
        _right = new bvh_node(hlist + n / 2, n - n / 2, rstate, level + 1);
        _right->set_id((level + 1) * 11) ;
    }

    aabb left_box, right_box;
    if (!_left->bounding_box(left_box) || 
        !_right->bounding_box(right_box)) {
        //printf("Error: No bounding box in bvh_node constructor\n");
    }
    _box = surrounding_box(left_box, right_box);
}

__device__ bool bvh_node::hit(const ray& r, float tmin, float tmax, hit_record& hrec) const {
    //return true;
    return _box.hit(r, tmin, tmax);
}

__device__ bool bvh_node::hit_shadow(const ray& r, float tmin, float tmax) const {
    return false;
}

__device__ bool bvh_node::dfs(const ray& r, float tmin, float tmax, hit_record& hrec) {
    if (!_box.hit(r, tmin, tmax)) return false;
    object* stack[STACK_SIZE];
    object** stack_ptr = stack;
    *stack_ptr = NULL; //stack bottom
    stack_ptr++;
    *stack_ptr = this;
    stack_ptr++;
    hit_record temp_rec;
    float closest = tmax;
    bool hit_anything = false;
   
    while (*--stack_ptr != NULL) {
        object* node = *stack_ptr;
        if (!node->is_leaf()) {
            if (node->hit(r, tmin, tmax, temp_rec)) {
                *stack_ptr++ = static_cast<bvh_node*>(node)->_left;
                *stack_ptr++ = static_cast<bvh_node*>(node)->_right;
            }
        } else {
            // leaf node; check if intersects
            if (node->hit(r, tmin, closest, temp_rec) && (temp_rec.t < closest)) {
                hit_anything = true;
                closest = temp_rec.t;
                hrec = temp_rec;
            }
        }
    }
    return hit_anything;
}

__device__ bool bvh_node::bounding_box(aabb& box) const {
    box = _box;
    return true;
}

// __device__ void bvh_node::display_tree(bvh_node* root, int level) {
//     printf("%*c|_", level, ' ');
//     printf("(%d) - <OBJECT_TYPE>\n", root->get_id()));
//     if (root->_left) {
//         if (!root->_left->is_leaf()) {
//             display_tree(static_cast<bvh_node*>(root->_left), level + 2);
//         } else {
//             printf("  %*c|_", level, ' ');
//             printf("(%d) LEFT %s (%.2f,%.2f,%.2f)\n", root->_left->get_id(), hitable_object::obj_type_str(root->_left->get_object_type()),
//                 static_cast<sphere*>(root->_left)->get_center().x(),
//                 static_cast<sphere*>(root->_left)->get_center().y(),
//                 static_cast<sphere*>(root->_left)->get_center().z());
//         }
//     }
//     if (root->_right) {
//         if (root->_right->get_object_type() == object_type::BOUNDING_VOLUME_HIERARCHY) {
//             display_tree(static_cast<bvh_node*>(root->_right), level + 2);
//         } else {
//             printf("  %*c|_", level, ' ');
//             printf("(%d) RIGHT %s (%.2f,%.2f,%.2f)\n", root->_right->get_id(), hitable_object::obj_type_str(root->_right->get_object_type()),
//                 static_cast<sphere*>(root->_left)->get_center().x(),
//                 static_cast<sphere*>(root->_left)->get_center().y(),
//                 static_cast<sphere*>(root->_left)->get_center().z());
//         }
//     }
// }
