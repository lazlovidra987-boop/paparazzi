/*
 * cv_ground_seg.c
 *
 * Ground segmentation for Paparazzi.
 *
 * Report-aligned logic:
 * 1. Interpret the incoming image as rotated 90 deg counterclockwise
 * 2. Downsample the logical image into blocks
 * 3. Classify each block in YUV422 using a small multi-sample vote
 * 4. Scan each logical column from bottom to top
 * 5. Reclassify short black gaps (< MIN_BLACK) back to ground
 * 6. Store visible ground depth in horizon[]
 * 7. Derive left / center / right scores
 * 8. Detect obstacle_ahead from middle columns
 *
 * Important notes:
 * - Gradient based edge detection is used to find the carptes
 * - Color detection is also used to find the trees
 * - Uses safer YUV422 access
 * - Uses explicit short-gap reclassification
 * - Keeps ABI interface compatible with existing setup
 * - Applies a hard 90 deg counterclockwise rotation in software so the
 *   segmentation logic sees the same orientation as intended
 */

#include "cv_ground_seg.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "std.h"

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>

#define GS_PRINT(string, ...) \
  fprintf(stderr, "[cv_ground_seg->%s()] " string, __FUNCTION__, ##__VA_ARGS__)

#ifndef GROUND_SEGMENTATION_FPS
#define GROUND_SEGMENTATION_FPS 0
#endif

#ifndef GS_DEBUG_EVERY_N_FRAMES
#define GS_DEBUG_EVERY_N_FRAMES 20
#endif

/* -------------------------------------------------------------------------- */
/* Internal shared state                                                      */
/* -------------------------------------------------------------------------- */

struct ground_seg_shared_t {
  struct ground_seg_result_t result;
  bool updated;
};

static pthread_mutex_t ground_seg_mutex;
static struct ground_seg_shared_t ground_seg_shared;
static uint32_t gs_frame_counter = 0U;

/* -------------------------------------------------------------------------- */
/* Tunable parameters                                                         */
/* -------------------------------------------------------------------------- */

uint8_t ground_lum_min = 50;
uint8_t ground_lum_max = 150;
uint8_t ground_cb_min  = 70;
uint8_t ground_cb_max  = 120;
uint8_t ground_cr_min  = 50;
uint8_t ground_cr_max  = 140;

uint8_t tree_lum_min = 0;
uint8_t tree_lum_max = 60;
uint8_t tree_cb_min  = 0;
uint8_t tree_cb_max  = 123;
uint8_t tree_cr_min  = 76;
uint8_t tree_cr_max  = 147;

uint8_t ground_downsize_x  = 4;
uint8_t ground_downsize_y  = 4;
uint8_t ground_min_black   = 5;
uint8_t ground_middle_cols = 10;

bool ground_draw = true;

/* Downsized binary classification map in LOGICAL rotated coordinates */
static uint8_t ground_small[GS_MAX_ROWS][GS_MAX_COLS];
static uint8_t carpet_small[GS_MAX_ROWS][GS_MAX_COLS];
static uint8_t tree_small[GS_MAX_ROWS][GS_MAX_COLS];

/* -------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* -------------------------------------------------------------------------- */

static void reset_result(struct ground_seg_result_t *res)
{
  memset(res, 0, sizeof(*res));
}

/*
 * Physical image dimensions from camera buffer:
 *   phys_w = img->w
 *   phys_h = img->h
 *
 * Logical image dimensions used by segmentation after 90 deg CCW rotation:
 *   logical_w = phys_h
 *   logical_h = phys_w
 */
static inline uint16_t logical_width(const struct image_t *img)
{
  return img->h;
}

static inline uint16_t logical_height(const struct image_t *img)
{
  return img->w;
}

/*
 * Map logical rotated coordinates to physical camera-buffer coordinates.
 *
 * 90 deg counterclockwise:
 *   src_x = phys_w - 1 - y_logical
 *   src_y = x_logical
 */
static inline void logical_to_physical(const struct image_t *img,
                                       uint16_t x_logical, uint16_t y_logical,
                                       uint16_t *x_phys, uint16_t *y_phys)
{
  uint16_t lw = logical_width(img);
  uint16_t lh = logical_height(img);

  if (x_logical >= lw) {
    x_logical = lw - 1U;
  }
  if (y_logical >= lh) {
    y_logical = lh - 1U;
  }

  *x_phys = (uint16_t)(img->w - 1U - y_logical);
  *y_phys = x_logical;
}

/*
 * Safe YUV422 reader in LOGICAL rotated coordinates.
 *
 * Assumed physical buffer layout:
 *   U Y0 V Y1
*/ 
static inline void get_yuv422_pixel(struct image_t *img, uint16_t x, uint16_t y,
                                    uint8_t *Y, uint8_t *U, uint8_t *V)
{
  if (img == NULL || img->buf == NULL || img->w < 2U || img->h == 0U) {
    *Y = 0U;
    *U = 0U;
    *V = 0U;
    return;
  }

  uint16_t src_x, src_y;
  logical_to_physical(img, x, y, &src_x, &src_y);

  uint16_t x_pair = (uint16_t)(src_x & ~1U);
  if (x_pair >= img->w - 1U) {
    x_pair = (img->w >= 2U) ? (img->w - 2U) : 0U;
  }

  uint32_t base = (uint32_t)src_y * 2U * img->w + 2U * x_pair;
  uint8_t *buffer = img->buf;

  uint8_t u  = buffer[base];
  uint8_t y0 = buffer[base + 1U];
  uint8_t v  = buffer[base + 2U];
  uint8_t y1 = buffer[base + 3U];

  *U = u;
  *V = v;
  *Y = ((src_x & 1U) == 0U) ? y0 : y1;
}

static inline bool is_ground_yuv(uint8_t Y, uint8_t U, uint8_t V)
{
  return (Y >= ground_lum_min && Y <= ground_lum_max &&
          U >= ground_cb_min  && U <= ground_cb_max  &&
          V >= ground_cr_min  && V <= ground_cr_max);
}

static inline bool is_tree_yuv(uint8_t Y, uint8_t U, uint8_t V)
{
  return (Y >= tree_lum_min && Y <= tree_lum_max &&
          U >= tree_cb_min  && U <= tree_cb_max  &&
          V >= tree_cr_min  && V <= tree_cr_max);
}


static uint8_t classify_block_vote(struct image_t *img,
                                   uint16_t x0, uint16_t y0,
                                   uint16_t block_w, uint16_t block_h)
{
  uint16_t lw = logical_width(img);
  uint16_t lh = logical_height(img);

  uint16_t x1 = (uint16_t)(x0 + block_w - 1U);
  uint16_t y1 = (uint16_t)(y0 + block_h - 1U);

  if (x1 >= lw) {
    x1 = lw - 1U;
  }
  if (y1 >= lh) {
    y1 = lh - 1U;
  }

  uint16_t xc = (uint16_t)((x0 + x1) / 2U);
  uint16_t yc = (uint16_t)((y0 + y1) / 2U);

  uint16_t xl = (uint16_t)((x0 + xc) / 2U);
  uint16_t xr = (uint16_t)((xc + x1) / 2U);
  uint16_t yu = (uint16_t)((y0 + yc) / 2U);
  uint16_t yd = (uint16_t)((yc + y1) / 2U);

  uint16_t sample_x[5] = { xc, xl, xr, xc, xc };
  uint16_t sample_y[5] = { yc, yc, yc, yu, yd };

  uint8_t hits = 0U;

  for (uint8_t i = 0U; i < 5U; i++) {
    uint8_t Y, U, V;
    get_yuv422_pixel(img, sample_x[i], sample_y[i], &Y, &U, &V);
    if (is_ground_yuv(Y, U, V)) {
      hits++;
    }
  }

  return (hits >= 3U) ? 1U : 0U;
}

static uint8_t classify_block_vote_tree(struct image_t *img,
                                   uint16_t x0, uint16_t y0,
                                   uint16_t block_w, uint16_t block_h)
{
  uint16_t lw = logical_width(img);
  uint16_t lh = logical_height(img);

  uint16_t x1 = (uint16_t)(x0 + block_w - 1U);
  uint16_t y1 = (uint16_t)(y0 + block_h - 1U);

  if (x1 >= lw) {
    x1 = lw - 1U;
  }
  if (y1 >= lh) {
    y1 = lh - 1U;
  }

  uint16_t xc = (uint16_t)((x0 + x1) / 2U);
  uint16_t yc = (uint16_t)((y0 + y1) / 2U);

  uint16_t xl = (uint16_t)((x0 + xc) / 2U);
  uint16_t xr = (uint16_t)((xc + x1) / 2U);
  uint16_t yu = (uint16_t)((y0 + yc) / 2U);
  uint16_t yd = (uint16_t)((yc + y1) / 2U);

  uint16_t sample_x[5] = { xc, xl, xr, xc, xc };
  uint16_t sample_y[5] = { yc, yc, yc, yu, yd };

  uint8_t hits = 0U;

  for (uint8_t i = 0U; i < 5U; i++) {
    uint8_t Y, U, V;
    get_yuv422_pixel(img, sample_x[i], sample_y[i], &Y, &U, &V);
    if (is_tree_yuv(Y, U, V)) {
      hits++;
    }
  }

  return (hits >= 3U) ? 1U : 0U;
}

/* -------------------------------------------------------------------------- */
/* Core processing                                                            */
/* -------------------------------------------------------------------------- */

static void build_ground_map(struct image_t *img, uint16_t cols, uint16_t rows)
{
  uint16_t lw = logical_width(img);
  uint16_t lh = logical_height(img);

  for (uint16_t r = 0U; r < rows; r++) {
    for (uint16_t c = 0U; c < cols; c++) {
      uint16_t x0 = (uint16_t)(c * ground_downsize_x);
      uint16_t y0 = (uint16_t)(r * ground_downsize_y);

      uint16_t block_w = ground_downsize_x;
      uint16_t block_h = ground_downsize_y;

      if (x0 >= lw) {
        x0 = lw - 1U;
      }
      if (y0 >= lh) {
        y0 = lh - 1U;
      }

      ground_small[r][c] = classify_block_vote(img, x0, y0, block_w, block_h);
      tree_small[r][c] = classify_block_vote_tree(img, x0, y0, block_w, block_h);

    }
  }
}

static void compute_horizon(struct ground_seg_result_t *res)
{
  uint16_t cols = res->cols;
  uint16_t rows = res->rows;

  for (uint16_t c = 0U; c < cols; c++) {
    uint16_t black_run = 0U;
    int16_t highest_visible_ground = -1;

    for (int16_t r = (int16_t)rows - 1; r >= 0; r--) {
      if (ground_small[r][c] != 0U) {
        if (black_run > 0U && black_run < ground_min_black) {
          for (uint16_t k = 1U; k <= black_run; k++) {
            uint16_t rr = (uint16_t)(r + (int16_t)k);
            if (rr < rows) {
              ground_small[rr][c] = 1U;
            }
          }
        }

        highest_visible_ground = r;
        black_run = 0U;
      } else {
        black_run++;
        if (black_run >= ground_min_black) {
          break;
        }
      }
    }

    if (highest_visible_ground < 0) {
      res->horizon[c] = 0U;
    } else {
      res->horizon[c] = (uint16_t)(rows - (uint16_t)highest_visible_ground);
    }
  }
}

static void compute_scores(struct ground_seg_result_t *res)
{
  uint16_t c1 = res->cols / 3U;
  uint16_t c2 = (2U * res->cols) / 3U;

  for (uint16_t c = 0U; c < res->cols; c++) {
    if (c < c1) {
      res->left_score += res->horizon[c];
    } else if (c < c2) {
      res->center_score += res->horizon[c];
    } else {
      res->right_score += res->horizon[c];
    }
  }
}

static void compute_obstacle_flag(struct ground_seg_result_t *res)
{
  res->obstacle_ahead = false;

  if (res->cols == 0U) {
    return;
  }

  uint16_t middle_cols = ground_middle_cols;
  if (middle_cols == 0U) {
    middle_cols = 1U;
  }
  if (middle_cols > res->cols) {
    middle_cols = res->cols;
  }

  uint16_t mid = res->cols / 2U;
  uint16_t half = middle_cols / 2U;

  uint16_t start = (mid > half) ? (uint16_t)(mid - half) : 0U;
  uint16_t end   = start + middle_cols;
  if (end > res->cols) {
    end = res->cols;
  }

  uint16_t open_cols = 0U;
  uint16_t total = 0U;

  for (uint16_t c = start; c < end; c++) {
    total++;
    if (res->horizon[c] > 1U) {
      open_cols++;
    }
  }

  if (total > 0U && open_cols + 1U < total) {
    res->obstacle_ahead = true;
  }
}

static void compute_centroid_and_ground_count(struct ground_seg_result_t *res)
{
  uint32_t total = 0U;
  uint32_t sum_x = 0U;
  uint32_t sum_y = 0U;

  for (uint16_t c = 0U; c < res->cols; c++) {
    uint16_t h = res->horizon[c];
    if (h == 0U) {
      continue;
    }

    total += h;
    sum_x += (uint32_t)c * h;

    uint16_t top_row = (uint16_t)(res->rows - h);
    uint16_t bottom_row = (uint16_t)(res->rows - 1U);
    uint16_t avg_row = (uint16_t)((top_row + bottom_row) / 2U);

    sum_y += (uint32_t)avg_row * h;
  }

  res->ground_count = total;

  if (total > 0U) {
    float cx = sum_x / (float)total;
    float cy = sum_y / (float)total;

    res->x_c = (int32_t)roundf(cx - (res->cols * 0.5f));
    res->y_c = (int32_t)roundf((res->rows * 0.5f) - cy);
  } else {
    res->x_c = 0;
    res->y_c = 0;
  }
}


/* -------------------------------------------------------------------------- */
/* Finding Carpets Core Logic                             */
/* -------------------------------------------------------------------------- */

#define CARPET_EDGE_THRESHOLD 150U
uint8_t dilation_iterations = 2;
uint8_t correction_iterations = 3;
uint8_t erode_again_iterations = 5;

static void dilate_obstacles(uint16_t cols, uint16_t rows, uint8_t iterations)
{
  for (uint8_t iter = 0U; iter < iterations; iter++) {
    /* Create a copy of ground_small to read from */
    uint8_t ground_temp[GS_MAX_ROWS][GS_MAX_COLS];
    memcpy(ground_temp, ground_small, sizeof(ground_small));

    /* Apply dilation: if a block is marked (0=obstacle), mark its neighbors */
    for (uint16_t r = 0U; r < rows; r++) {
      for (uint16_t c = 0U; c < cols; c++) {
        if (ground_temp[r][c] == 0U) {
          /* Mark all 8 neighbors */
          if (c > 0U) {
            ground_small[r][c - 1U] = 0U;  /* Left */
          }
          if (c < cols - 1U) {
            ground_small[r][c + 1U] = 0U;  /* Right */
          }
          if (r > 0U) {
            ground_small[r - 1U][c] = 0U;  /* Up */
          }
          if (r < rows - 1U) {
            ground_small[r + 1U][c] = 0U;  /* Down */
          }
          /* Diagonals */
          if (r > 0U && c > 0U) {
            ground_small[r - 1U][c - 1U] = 0U;
          }
          if (r > 0U && c < cols - 1U) {
            ground_small[r - 1U][c + 1U] = 0U;
          }
          if (r < rows - 1U && c > 0U) {
            ground_small[r + 1U][c - 1U] = 0U;
          }
          if (r < rows - 1U && c < cols - 1U) {
            ground_small[r + 1U][c + 1U] = 0U;
          }
        }
      }
    }
  }
}

static void correct_carpets(uint16_t cols, uint16_t rows, uint8_t iterations)
{
 /* Define qué consideramos "suficientemente rodeado". 
   * 5 significa que al menos 5 de las 8 celdas vecinas deben ser 1. */
  const uint8_t THRESHOLD = 5U; 

  for (uint8_t iter = 0U; iter < iterations; iter++) {
    /* Crear una copia para lectura y evitar modificar la matriz mientras la evaluamos */
    uint8_t carpet_temp[GS_MAX_ROWS][GS_MAX_COLS];
    memcpy(carpet_temp, carpet_small, sizeof(carpet_small));

    for (uint16_t r = 0U; r < rows; r++) {
      for (uint16_t c = 0U; c < cols; c++) {
        
        /* Evaluar solo las celdas que son 0 */
        if (carpet_temp[r][c] == 0U) {
          uint8_t count_ones = 0U;

          /* Contar los vecinos en las 8 direcciones */
          if (c > 0U && carpet_temp[r][c - 1U] == 1U) count_ones++;                 /* Izquierda */
          if (c < cols - 1U && carpet_temp[r][c + 1U] == 1U) count_ones++;          /* Derecha */
          if (r > 0U && carpet_temp[r - 1U][c] == 1U) count_ones++;                 /* Arriba */
          if (r < rows - 1U && carpet_temp[r + 1U][c] == 1U) count_ones++;          /* Abajo */
          if (r > 0U && c > 0U && carpet_temp[r - 1U][c - 1U] == 1U) count_ones++;  /* Arriba-Izquierda */
          if (r > 0U && c < cols - 1U && carpet_temp[r - 1U][c + 1U] == 1U) count_ones++; /* Arriba-Derecha */
          if (r < rows - 1U && c > 0U && carpet_temp[r + 1U][c - 1U] == 1U) count_ones++; /* Abajo-Izquierda */
          if (r < rows - 1U && c < cols - 1U && carpet_temp[r + 1U][c + 1U] == 1U) count_ones++; /* Abajo-Derecha */

          /* Si la celda 0 está suficientemente rodeada de 1s, se convierte en 1 */
          if (count_ones >= THRESHOLD) {
            carpet_small[r][c] = 1U;
          }
        }
      }
    }
  }
}

static void correct_carpets_again(uint16_t cols, uint16_t rows, uint8_t iterations)
{
 /* Define qué consideramos "suficientemente rodeado". 
   * 5 significa que al menos 5 de las 8 celdas vecinas deben ser 1. */
  const uint8_t THRESHOLD = 5U; 

  for (uint8_t iter = 0U; iter < iterations; iter++) {
    /* Crear una copia para lectura y evitar modificar la matriz mientras la evaluamos */
    uint8_t carpet_temp[GS_MAX_ROWS][GS_MAX_COLS];
    memcpy(carpet_temp, carpet_small, sizeof(carpet_small));

    for (uint16_t r = 0U; r < rows; r++) {
      for (uint16_t c = 0U; c < cols; c++) {
        
        /* Evaluar solo las celdas que son 0 */
        if (carpet_temp[r][c] == 1U) {
          uint8_t count_ones = 0U;

          /* Contar los vecinos en las 8 direcciones */
          if (c > 0U && carpet_temp[r][c - 1U] == 0U) count_ones++;                 /* Izquierda */
          if (c < cols - 1U && carpet_temp[r][c + 1U] == 0U) count_ones++;          /* Derecha */
          if (r > 0U && carpet_temp[r - 1U][c] == 0U) count_ones++;                 /* Arriba */
          if (r < rows - 1U && carpet_temp[r + 1U][c] == 0U) count_ones++;          /* Abajo */
          if (r > 0U && c > 0U && carpet_temp[r - 1U][c - 1U] == 0U) count_ones++;  /* Arriba-Izquierda */
          if (r > 0U && c < cols - 1U && carpet_temp[r - 1U][c + 1U] == 0U) count_ones++; /* Arriba-Derecha */
          if (r < rows - 1U && c > 0U && carpet_temp[r + 1U][c - 1U] == 0U) count_ones++; /* Abajo-Izquierda */
          if (r < rows - 1U && c < cols - 1U && carpet_temp[r + 1U][c + 1U] == 0U) count_ones++; /* Abajo-Derecha */

          /* Si la celda 0 está suficientemente rodeada de 1s, se convierte en 1 */
          if (count_ones >= THRESHOLD) {
            carpet_small[r][c] = 0U;
          }
        }
      }
    }
  }
}

static void find_carpet(struct image_t *img, uint16_t cols, uint16_t rows)
{
  /* 1. Limpiamos la matriz global de alfombras por si quedó algo del fotograma anterior */
  memset(carpet_small, 0, sizeof(carpet_small));

  // uint16_t row_start = (uint16_t)((2U * rows) / 3U);
  uint16_t row_start = (uint16_t)((2U * rows) / 3U);
  /* 2. Recorremos la cuadrícula matemática (igual que build_ground_map) */
  for (uint16_t r = row_start; r < rows; r++) {
    for (uint16_t c = 0U; c < cols; c++) {
      
      uint16_t x0 = (uint16_t)(c * ground_downsize_x);
      uint16_t y0 = (uint16_t)(r * ground_downsize_y);

      /* With image padding, we can now process blocks at the edges.
         The safety margin that was needed for the original image is now 
         handled by the padded pixels. */
      uint16_t max_x = logical_width(img);
      uint16_t max_y = logical_height(img);
      
      if (x0 >= max_x || y0 >= max_y) {
        continue; /* Only skip if completely out of bounds */
      }

      uint32_t total_edge_magnitude = 0U;

      /* 3. Escaneamos los píxeles DENTRO de este bloque específico */
      for (uint16_t y = y0; y < y0 + ground_downsize_y && y < max_y; y++) {
        for (uint16_t x = x0; x < x0 + ground_downsize_x && x < max_x; x++) {
          
          uint8_t Y_left, Y_right, Y_up, Y_down, U, V;

          /* Usamos la función nativa del dron para leer la Luminancia (Y) de los vecinos.
             (Ignoramos U y V porque los bordes se detectan mejor en blanco y negro) */
          get_yuv422_pixel(img, x - 1U, y, &Y_left,  &U, &V);
          get_yuv422_pixel(img, x + 1U, y, &Y_right, &U, &V);
          get_yuv422_pixel(img, x, y - 1U, &Y_up,    &U, &V);
          get_yuv422_pixel(img, x, y + 1U, &Y_down,  &U, &V);

          /* 4. Aproximación rápida de gradiente (Gx y Gy) */
          int16_t Gx = (int16_t)Y_right - (int16_t)Y_left;
          int16_t Gy = (int16_t)Y_down - (int16_t)Y_up;

          /* La magnitud del borde es la suma absoluta de los cambios en X e Y */
          total_edge_magnitude += (uint32_t)(abs(Gx) + abs(Gy));
        }
      }

      /* 5. Decisión binaria: ¿Este bloque tiene suficientes bordes para ser alfombra? */
      if (total_edge_magnitude > CARPET_EDGE_THRESHOLD) {
        carpet_small[r][c] = 1U; /* ¡Es alfombra! */
      } 
      // Nota: No hace falta poner = 0U porque ya hicimos el memset a cero arriba
    }
  }
}

/* -------------------------------------------------------------------------- */
/* Main analysis                                                              */
/* -------------------------------------------------------------------------- */

static uint32_t ground_seg_analyse_image(struct image_t *img,
                                         struct ground_seg_result_t *res)
{
  reset_result(res);

  if (img == NULL || img->buf == NULL) {
    return 0U;
  }

  if (ground_downsize_x == 0U || ground_downsize_y == 0U) {
    return 0U;
  }

  uint16_t lw = logical_width(img);
  uint16_t lh = logical_height(img);

  uint16_t cols = lw / ground_downsize_x;
  uint16_t rows = lh / ground_downsize_y;

  if (cols == 0U || rows == 0U) {
    return 0U;
  }

  if (cols > GS_MAX_COLS) {
    cols = GS_MAX_COLS;
  }
  if (rows > GS_MAX_ROWS) {
    rows = GS_MAX_ROWS;
  }

  res->cols = cols;
  res->rows = rows;

  find_carpet(img, cols, rows);
  correct_carpets(cols, rows, correction_iterations);
  correct_carpets_again(cols, rows, erode_again_iterations);

  build_ground_map(img, cols, rows);

  for (uint16_t r = 0U; r < rows; r++) {
    for (uint16_t c = 0U; c < cols; c++) {
      ground_small[r][c] = ground_small[r][c] | carpet_small[r][c];
      ground_small[r][c] = ground_small[r][c] & ~tree_small[r][c]; // Tree detected (tree = 1)
    }
  }

  dilate_obstacles(cols, rows, dilation_iterations);

  compute_horizon(res);

  compute_scores(res);
  compute_obstacle_flag(res);
  compute_centroid_and_ground_count(res);

  return res->ground_count;
}

/* -------------------------------------------------------------------------- */
/* Paparazzi hooks                                                            */
/* -------------------------------------------------------------------------- */

static struct image_t *ground_seg_process_image(struct image_t *img, uint8_t camera_id)
{
  (void)camera_id;

  struct ground_seg_result_t local_result;
  ground_seg_analyse_image(img, &local_result);

  pthread_mutex_lock(&ground_seg_mutex);
  memcpy(&ground_seg_shared.result, &local_result, sizeof(local_result));
  ground_seg_shared.updated = true;
  pthread_mutex_unlock(&ground_seg_mutex);

  return img;
}

void ground_segmentation_init(void)
{
  memset(&ground_seg_shared, 0, sizeof(ground_seg_shared));
  pthread_mutex_init(&ground_seg_mutex, NULL);
  gs_frame_counter = 0U;

#ifdef GROUND_SEGMENTATION_DRAW
  ground_draw = GROUND_SEGMENTATION_DRAW;
#endif

#ifdef GROUND_SEGMENTATION_CAMERA
  cv_add_to_device(&GROUND_SEGMENTATION_CAMERA,
                   ground_seg_process_image,
                   GROUND_SEGMENTATION_FPS,
                   0);
#endif

}

void ground_segmentation_periodic(void)
{
  struct ground_seg_result_t local_result;
  bool updated = false;

  pthread_mutex_lock(&ground_seg_mutex);
  memcpy(&local_result, &ground_seg_shared.result, sizeof(local_result));
  updated = ground_seg_shared.updated;
  ground_seg_shared.updated = false;
  pthread_mutex_unlock(&ground_seg_mutex);

  if (!updated) {
    return;
  }
}

void ground_seg_get_result(struct ground_seg_result_t *out)
{
  pthread_mutex_lock(&ground_seg_mutex);
  memcpy(out, &ground_seg_shared.result, sizeof(*out));
  pthread_mutex_unlock(&ground_seg_mutex);
}