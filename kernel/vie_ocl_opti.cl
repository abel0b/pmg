#include "kernel/common.cl"

#define img_cell(i, l, c) ((i) + (l) * DIM + (c))
#define cur_img(y, x) (*img_cell (image, (y), (x)))
#define next_img(y, x) (*img_cell (alt_image, (y), (x)))
#define change(y, x) (*img_cell (changes, (y+1), (x+1)))

__constant unsigned rules[2][9] = {
    {0,0,0,0xFFFF00FF,0,0,0,0,0},
    {0,0,0xFFFF00FF,0xFFFF00FF,0,0,0,0,0},
};

__kernel void vie(__global unsigned * image, __global unsigned * alt_image, __global unsigned * changes) {
    unsigned x = get_global_id(0);
    unsigned y = get_global_id(1);
    unsigned tilex = get_local_id(0);
    unsigned tiley = get_local_id(1);

    if (change(tiley, tilex)) {
        unsigned n = (cur_img(y-1, x-1) != 0) + (cur_img(y-1, x) != 0) + (cur_img(y-1, x+1) != 0) + (cur_img(y, x-1) != 0) + (cur_img(y, x+1) != 0) + (cur_img(y+1, x-1) != 0) + (cur_img(y+1, x) != 0) + (cur_img(y+1, x+1) != 0);
        unsigned alive = cur_img (y, x) != 0;
        next_img (y, x) = rules[alive][n];
    }
}
