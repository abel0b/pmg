#include "kernel/common.cl"

#define img_cell(i, l, c) ((i) + (l) * DIM + (c))
#define cur_img(y, x) (*img_cell (image, (y), (x)))
#define next_img(y, x) (*img_cell (alt_image, (y), (x)))
#define change(y, x) (*img_cell (changes, (y+1), (x+1)))

typedef Uint32 cell_t;

#define OPTI

__constant cell_t rules[2][9] = {
    {0,0,0,0xFFFF00FF,0,0,0,0,0},
    {0,0,0xFFFF00FF,0xFFFF00FF,0,0,0,0,0},
};

__constant unsigned rules_change[2][9] = {
    {0,0,0,1,0,0,0,0,0},
    {1,1,0,0,1,1,1,1,1},
};

__kernel void vie(__global cell_t * image, __global cell_t * alt_image, __global cell_t * changes) {
    unsigned x = get_global_id(0);
    unsigned y = get_global_id(1);
    unsigned tilex = get_local_id(0);
    unsigned tiley = get_local_id(1);

    int need_compute = cur_changes(tiley, tilex) | cur_changes(tiley-1, tilex-1) | cur_changes(tiley, tilex-1) | cur_changes(tiley+1, tilex-1) | cur_changes(tiley-1, tilex) | cur_changes(tiley+1, tilex) | cur_changes(tiley-1, tilex+1) | cur_changes(tiley, tilex+1) | cur_changes(tiley+1, tilex+1);

    if (need_compute) {
        barrier(CLK_LOCAL_MEM_FENCE);
        unsigned n = (cur_img(y-1, x-1) != 0) + (cur_img(y-1, x) != 0) + (cur_img(y-1, x+1) != 0) + (cur_img(y, x-1) != 0) + (cur_img(y, x+1) != 0) + (cur_img(y+1, x-1) != 0) + (cur_img(y+1, x) != 0) + (cur_img(y+1, x+1) != 0);
        unsigned alive = cur_img (y, x) != 0;
        next_img (y, x) = rules[alive][n];
        change(tiley,tilex) = change(tiley,tilex) | rules_change[alive][n];
    }
}
