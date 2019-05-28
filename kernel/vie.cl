unsigned rules_change[2][9] = {
    {0,0,0,1,0,0,0,0,0},
    {1,1,0,0,1,1,1,1,1},
};

unsigned rules_image[2][9] = {
    {0,0,0,0xFFFF00FF,0,0,0,0,0},
    {0,0,0xFFFF00FF,0xFFFF00FF,0,0,0,0,0},
};

static inline Uint32 *img_cell (Uint32 *i, int l, int c)
{
  return i + l * DIM + c;
}

#define cur_img(y, x) (*img_cell (image, (y), (x)))
#define next_img(y, x) (*img_cell (alt_image, (y), (x)))

int compute_new_state (int y, int x) {
  unsigned n      = 0;
  unsigned change = 0;
  unsigned alive = 0;

  if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1) {
    n = (cur_img(y-1, x-1) != 0) + (cur_img(y-1, x) != 0) + (cur_img(y-1, x+1) != 0) + (cur_img(y, x-1) != 0) + (cur_img(y, x+1) != 0) + (cur_img(y+1, x-1) != 0) + (cur_img(y+1, x) != 0) + (cur_img(y+1, x+1) != 0);

    alive = cur_img (y, x) != 0;
    change = rules_change[alive][n];
    next_img (y, x) = rules_image[alive][n];
  }

  return change;
}

__kernel void vie(__global Uint32 * image, __global Uint32 * alt_image) {
    unsigned x = get_global_id(0);
    unsigned y = get_global_id(1);
    compute_new_state(y, x);
}
