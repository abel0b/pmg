
#ifndef GRAPHICS_IS_DEF
#define GRAPHICS_IS_DEF

#ifndef NOSDL
#include <SDL.h>
#else
#include <stdint.h>
typedef uint32_t Uint32;
typedef uint8_t Uint8;
#endif

#include "global.h"

void graphics_init ();
void graphics_share_texture_buffers (void);
void graphics_refresh (void);
void graphics_dump_image_to_file (char *filename);
void graphics_clean (void);
int graphics_display_enabled (void);

typedef Uint32 cell_t;

extern cell_t *restrict image, *restrict alt_image;

static inline cell_t *img_cell (Uint32 *i, int l, int c)
{
  return i + l * DIM + c;
}

#define cur_img(y, x) (*img_cell (image, (y), (x)))
#define next_img(y, x) (*img_cell (alt_image, (y), (x)))

static inline void swap_images (void)
{
  cell_t *tmp = image;

  image     = alt_image;
  alt_image = tmp;
}

#endif
