
#include "compute.h"
#include "debug.h"
#include "global.h"
#include "graphics.h"
#include "ocl.h"
#include "scheduler.h"
#include "monitoring.h"

#include <stdbool.h>
#include <omp.h>

char ** changes = NULL;
char ** next_changes = NULL;

void vie_init() {
  changes = malloc((GRAIN+2) * sizeof(bool *));
  next_changes = malloc((GRAIN+2) * sizeof(bool *));
  for (int i=0; i<(GRAIN+2); i++) {
    changes[i] = malloc((GRAIN+2) * sizeof(bool));
    next_changes[i] = malloc((GRAIN+2) * sizeof(bool));
    if (i == 0 || i == GRAIN+1) {
      memset(changes[i], 0, (GRAIN+2));
    }
    else {
      memset(changes[i] + sizeof(char), 1, GRAIN);
      changes[i][0] = 0;
      changes[i][GRAIN+1] = 0;
    }
  }
}

void vie_finalize() {
  free(changes);
  free(next_changes);
}

// void vie_refresh_img() {
//
// }

char need_compute(int tilex, int tiley) {
  tilex++;
  tiley++;
  return changes[tilex][tiley] || changes[tilex-1][tiley-1] || changes[tilex-1][tiley] || changes[tilex-1][tiley+1] || changes[tilex][tiley-1] || changes[tilex][tiley+1] || changes[tilex+1][tiley-1] || changes[tilex+1][tiley] || changes[tilex+1][tiley+1];
}

unsigned rules_change[2][9] = {
    {0,0,0,1,0,0,0,0,0},
    {1,1,0,0,1,1,1,1,1},
};

unsigned rules_image[2][9] = {
    {0,0,0,0xFFFF00FF,0,0,0,0,0},
    {0,0,0xFFFF00FF,0xFFFF00FF,0,0,0,0,0},
};

static int compute_new_state (int y, int x)
{
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

static int traiter_tuile (int i_d, int j_d, int i_f, int j_f)
{
  unsigned change = 0;

  #ifdef ENABLE_MONITORING
          monitoring_add_tile (i_d, j_d, DIM/GRAIN, DIM/GRAIN,
                               omp_get_thread_num ());
  #endif

  PRINT_DEBUG ('c', "tuile [%d-%d][%d-%d] traitée\n", i_d, i_f, j_d, j_f);

  for (int i = i_d; i <= i_f; i++)
    for (int j = j_d; j <= j_f; j++) {
      change |= compute_new_state (i, j);
    }

  return change;
}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned vie_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    // On traite toute l'image en un coup (oui, c'est une grosse tuile)
    unsigned change = traiter_tuile (0, 0, DIM - 1, DIM - 1);

    swap_images ();

    if (!change)
      return it;
  }

  return 0;
}

unsigned vie_compute_seq_tile (unsigned nb_iter)
{
  unsigned change;
  for (unsigned it = 1; it <= nb_iter; it++) {
    change = 0;
    for (int tilex=0; tilex<GRAIN; tilex++) {
      for (int tiley=0; tiley<GRAIN; tiley++) {
        change |= traiter_tuile (tilex*(DIM / GRAIN), tiley*(DIM / GRAIN), (tilex+1)*(DIM / GRAIN)-1, (tiley+1)*(DIM / GRAIN)-1);
      }
    }
    swap_images ();

    if (!change)
      return it;
  }

  return 0;
}

unsigned vie_compute_seq_opti (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (int tilex=0; tilex<GRAIN; tilex++) {
      for (int tiley=0; tiley<GRAIN; tiley++) {
        if (need_compute(tilex, tiley)) {
          next_changes[tilex+1][tiley+1] = traiter_tuile (tilex*(DIM / GRAIN), tiley*(DIM / GRAIN), (tilex+1)*(DIM / GRAIN)-1, (tiley+1)*(DIM / GRAIN)-1)? 1 : 0;
        }
      }
    }

    void * tmp_change = changes;
    changes = next_changes;
    next_changes = tmp_change;

    swap_images ();
  }

  return 0;
}

unsigned vie_compute_omp (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= DIM-1; i++) {
      for (int j = 0; j <= DIM-1; j++) {
        compute_new_state (i, j);
      }
    }

    swap_images ();
  }

  return 0;
}

unsigned vie_compute_omp_tile (unsigned nb_iter)
{
  unsigned change;
  for (unsigned it = 1; it <= nb_iter; it++) {
    change = 0;
    #pragma omp parallel for collapse(2)
    for (int tilex=0; tilex<GRAIN; tilex++) {
      for (int tiley=0; tiley<GRAIN; tiley++) {
        change |= traiter_tuile (tilex*(DIM / GRAIN), tiley*(DIM / GRAIN), (tilex+1)*(DIM / GRAIN)-1, (tiley+1)*(DIM / GRAIN)-1);
      }
    }
    swap_images ();

    if (!change)
      return it;
  }

  return 0;
}

unsigned vie_compute_omp_opti (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {
    #pragma omp parallel for collapse(2)
    for (int tilex=0; tilex<GRAIN; tilex++) {
      for (int tiley=0; tiley<GRAIN; tiley++) {
        if (need_compute(tilex, tiley)) {
          next_changes[tilex+1][tiley+1] = (traiter_tuile (tilex*(DIM / GRAIN), tiley*(DIM / GRAIN), (tilex+1)*(DIM / GRAIN)-1, (tiley+1)*(DIM / GRAIN)-1))? 1 : 0;
        }
      }
    }

    void * tmp_change = changes;
    changes = next_changes;
    next_changes = tmp_change;
    swap_images ();
  }

  return 0;
}

unsigned vie_compute_omp_task_tile (unsigned nb_iter)
{
  unsigned change;
  for (unsigned it = 1; it <= nb_iter; it++) {
    change = 0;
    #pragma omp parallel
    {
      #pragma omp single
      {
        for (int tilex=0; tilex<GRAIN; tilex++) {
          for (int tiley=0; tiley<GRAIN; tiley++) {
            #pragma omp task
            change |= traiter_tuile (tilex*(DIM / GRAIN), tiley*(DIM / GRAIN), (tilex+1)*(DIM / GRAIN)-1, (tiley+1)*(DIM / GRAIN)-1);
          }
        }
      }
    }
    swap_images ();

    if (!change)
      return it;
  }

  return 0;
}

unsigned vie_compute_omp_task_opti (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {
    #pragma omp parallel
    #pragma omp single
    for (int tilex=0; tilex<GRAIN; tilex++) {
      for (int tiley=0; tiley<GRAIN; tiley++) {
        #pragma omp task
        if (need_compute(tilex, tiley)) {
          next_changes[tilex+1][tiley+1] = (traiter_tuile (tilex*(DIM / GRAIN), tiley*(DIM / GRAIN), (tilex+1)*(DIM / GRAIN)-1, (tiley+1)*(DIM / GRAIN)-1))? 1 : 0;
        }
      }
    }


    #pragma omp parallel for
    for (int i=0; i<GRAIN; i++) {
        memcpy(changes[i+1] + sizeof(char), next_changes[i+1] + sizeof(char), GRAIN);
    }
    swap_images ();
  }

  return 0;
}


///////////////////////////// Configuration initiale

void draw_stable (void);
void draw_guns (void);
void draw_random (void);
void draw_clown (void);
void draw_diehard (void);

void vie_draw (char *param)
{
  char func_name[1024];
  void (*f) (void) = NULL;

  if (param == NULL)
    f = draw_guns;
  else {
    sprintf (func_name, "draw_%s", param);
    f = dlsym (DLSYM_FLAG, func_name);

    if (f == NULL) {
      PRINT_DEBUG ('g', "Cannot resolve draw function: %s\n", func_name);
      f = draw_guns;
    }
  }

  f ();
}

unsigned vie_compute_ocl (unsigned nb_iter)
{
  size_t global[2] = {SIZE, SIZE};   // global domain size for our calculation
  size_t local[2]  = {TILEX, TILEY}; // local domain size for our calculation
  cl_int err;

  for (unsigned it = 1; it <= nb_iter; it++) {
    // Set kernel arguments
    //
    err = 0;
    err |= clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &cur_buffer);
    err |= clSetKernelArg (compute_kernel, 1, sizeof (Uint32*), image);
    err |= clSetKernelArg (compute_kernel, 2, sizeof (Uint32*), alt_image);

    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local,
                                  0, NULL, NULL);
    check (err, "Failed to execute kernel");

  }

  return 0;
}

static unsigned couleur = 0xFFFF00FF; // Yellow

static void gun (int x, int y, int version)
{
  bool glider_gun[11][38] = {
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
       0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0},
      {0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
       0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1,
       0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  };

  if (version == 0)
    for (int i = 0; i < 11; i++)
      for (int j = 0; j < 38; j++)
        if (glider_gun[i][j])
          cur_img (i + x, j + y) = couleur;

  if (version == 1)
    for (int i = 0; i < 11; i++)
      for (int j = 0; j < 38; j++)
        if (glider_gun[i][j])
          cur_img (x - i, j + y) = couleur;

  if (version == 2)
    for (int i = 0; i < 11; i++)
      for (int j = 0; j < 38; j++)
        if (glider_gun[i][j])
          cur_img (x - i, y - j) = couleur;

  if (version == 3)
    for (int i = 0; i < 11; i++)
      for (int j = 0; j < 38; j++)
        if (glider_gun[i][j])
          cur_img (i + x, y - j) = couleur;
}

void draw_stable (void)
{
  for (int i = 1; i < DIM - 2; i += 4)
    for (int j = 1; j < DIM - 2; j += 4)
      cur_img (i, j) = cur_img (i, (j + 1)) = cur_img ((i + 1), j) =
          cur_img ((i + 1), (j + 1))        = couleur;
}

void draw_guns (void)
{
  memset (&cur_img (0, 0), 0, DIM * DIM * sizeof (cur_img (0, 0)));

  gun (0, 0, 0);
  gun (0, DIM - 1, 3);
  gun (DIM - 1, DIM - 1, 2);
  gun (DIM - 1, 0, 1);
}

void draw_random (void)
{
  for (int i = 1; i < DIM - 1; i++)
    for (int j = 1; j < DIM - 1; j++)
      cur_img (i, j) = random () & 01;
}

void draw_clown (void)
{
  memset (&cur_img (0, 0), 0, DIM * DIM * sizeof (cur_img (0, 0)));

  int mid                = DIM / 2;
  cur_img (mid, mid - 1) = cur_img (mid, mid) = cur_img (mid, mid + 1) =
      couleur;
  cur_img (mid + 1, mid - 1) = cur_img (mid + 1, mid + 1) = couleur;
  cur_img (mid + 2, mid - 1) = cur_img (mid + 2, mid + 1) = couleur;
}

void draw_diehard (void)
{
  memset (&cur_img (0, 0), 0, DIM * DIM * sizeof (cur_img (0, 0)));

  int mid = DIM / 2;

  cur_img (mid, mid - 3) = cur_img (mid, mid - 2) = couleur;
  cur_img (mid + 1, mid - 2)                      = couleur;

  cur_img (mid - 1, mid + 3)     = couleur;
  cur_img (mid + 1, mid + 2)     = cur_img (mid + 1, mid + 3) =
      cur_img (mid + 1, mid + 4) = couleur;
}
