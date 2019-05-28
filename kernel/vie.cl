#include "kernel/common.cl"

#define img_cell(i, l, c) ((i) + (l) * DIM + (c))
#define cur_img(y, x) (*img_cell (image, (y), (x)))
#define next_img(y, x) (*img_cell (alt_image, (y), (x)))

__kernel void vie(__global Uint32 * image, __global Uint32 * alt_image, unsigned rules[2][9]) {
    unsigned x = get_global_id(0);
    unsigned y = get_global_id(1);

    unsigned n = (cur_img(y-1, x-1) != 0) + (cur_img(y-1, x) != 0) + (cur_img(y-1, x+1) != 0) + (cur_img(y, x-1) != 0) + (cur_img(y, x+1) != 0) + (cur_img(y+1, x-1) != 0) + (cur_img(y+1, x) != 0) + (cur_img(y+1, x+1) != 0);
    unsigned alive = cur_img (y, x) != 0;
    next_img (y, x) = rules[alive][n];
}
