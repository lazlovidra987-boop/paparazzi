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

/* Downsized binary classification maps in LOGICAL rotated coordinates */
static uint8_t ground_small[GS_MAX_ROWS][GS_MAX_COLS];
static uint8_t carpet_small[GS_MAX_ROWS][GS_MAX_COLS];
static uint8_t tree_small[GS_MAX_ROWS][GS_MAX_COLS];

static uint8_t ground_temp[GS_MAX_ROWS][GS_MAX_COLS];
static uint8_t carpet_temp[GS_MAX_ROWS][GS_MAX_COLS];

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

static inline void get_yuv422_pixel_fast(const uint8_t *buf,
                                         uint16_t img_w,
                                         uint16_t src_x, uint16_t src_y,
                                         uint8_t *Y, uint8_t *U, uint8_t *V)
{
  uint16_t x_pair = (uint16_t)(src_x & ~1U);
  if (x_pair >= img_w - 1U) {
    x_pair = img_w - 2U;
  }
  uint32_t base = (uint32_t)src_y * 2U * img_w + 2U * x_pair;
  *U = buf[base];
  *V = buf[base + 2U];
  *Y = ((src_x & 1U) == 0U) ? buf[base + 1U] : buf[base + 3U];
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

static void classify_block_vote_both(struct image_t *img,
                                     uint16_t x0, uint16_t y0,
                                     uint16_t block_w, uint16_t block_h,
                                     uint8_t *out_ground, uint8_t *out_tree)
{
  const uint16_t lw = logical_width(img);
  const uint16_t lh = logical_height(img);
  const uint8_t  *buf = (const uint8_t *)img->buf;
  const uint16_t  img_w = img->w;

  uint16_t x1 = (uint16_t)(x0 + block_w - 1U);
  uint16_t y1 = (uint16_t)(y0 + block_h - 1U);
  if (x1 >= lw) { x1 = lw - 1U; }
  if (y1 >= lh) { y1 = lh - 1U; }

  const uint16_t xc = (uint16_t)((x0 + x1) / 2U);
  const uint16_t yc = (uint16_t)((y0 + y1) / 2U);
  const uint16_t xl = (uint16_t)((x0 + xc) / 2U);
  const uint16_t xr = (uint16_t)((xc + x1) / 2U);
  const uint16_t yu = (uint16_t)((y0 + yc) / 2U);
  const uint16_t yd = (uint16_t)((yc + y1) / 2U);

  const uint16_t sample_x[5] = { xc, xl, xr, xc, xc };
  const uint16_t sample_y[5] = { yc, yc, yc, yu, yd };

  uint8_t ground_hits = 0U;
  uint8_t tree_hits   = 0U;

  for (uint8_t i = 0U; i < 5U; i++) {
    uint16_t lx = sample_x[i];
    uint16_t ly = sample_y[i];
    if (lx >= lw) { lx = lw - 1U; }
    if (ly >= lh) { ly = lh - 1U; }
    uint16_t sx = (uint16_t)(img_w - 1U - ly);
    uint16_t sy = lx;

    uint8_t Y, U, V;
    get_yuv422_pixel_fast(buf, img_w, sx, sy, &Y, &U, &V);

    if (is_ground_yuv(Y, U, V)) { ground_hits++; }
    if (is_tree_yuv(Y, U, V))   { tree_hits++;   }
  }

  *out_ground = (ground_hits >= 3U) ? 1U : 0U;
  *out_tree   = (tree_hits   >= 3U) ? 1U : 0U;
}

/* -------------------------------------------------------------------------- */
/* Core processing                                                            */
/* -------------------------------------------------------------------------- */

static void build_ground_map(struct image_t *img, uint16_t cols, uint16_t rows)
{
  const uint16_t lw = logical_width(img);
  const uint16_t lh = logical_height(img);

  for (uint16_t r = 0U; r < rows; r++) {
    for (uint16_t c = 0U; c < cols; c++) {
      uint16_t x0 = (uint16_t)(c * ground_downsize_x);
      uint16_t y0 = (uint16_t)(r * ground_downsize_y);

      if (x0 >= lw) { x0 = lw - 1U; }
      if (y0 >= lh) { y0 = lh - 1U; }

      classify_block_vote_both(img, x0, y0,
                               ground_downsize_x, ground_downsize_y,
                               &ground_small[r][c],
                               &tree_small[r][c]);
    }
  }
}

static void compute_horizon(struct ground_seg_result_t *res)
{
  const uint16_t cols = res->cols;
  const uint16_t rows = res->rows;

  for (uint16_t c = 0U; c < cols; c++) {
    uint16_t black_run = 0U;
    int16_t  highest_visible_ground = -1;

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
        if (black_run >= ground_min_black) { break; }
      }
    }

    res->horizon[c] = (highest_visible_ground < 0)
                      ? 0U
                      : (uint16_t)(rows - (uint16_t)highest_visible_ground);
  }
}

static void compute_scores(struct ground_seg_result_t *res)
{
  const uint16_t c1 = res->cols / 3U;
  const uint16_t c2 = (2U * res->cols) / 3U;

  for (uint16_t c = 0U; c < res->cols; c++) {
    if (c < c1) {
      res->left_score   += res->horizon[c];
    } else if (c < c2) {
      res->center_score += res->horizon[c];
    } else {
      res->right_score  += res->horizon[c];
    }
  }
}

static void compute_obstacle_flag(struct ground_seg_result_t *res)
{
  res->obstacle_ahead = false;

  if (res->cols == 0U) { return; }

  uint16_t middle_cols = ground_middle_cols;
  if (middle_cols == 0U)        { middle_cols = 1U; }
  if (middle_cols > res->cols)  { middle_cols = res->cols; }

  const uint16_t mid  = res->cols / 2U;
  const uint16_t half = middle_cols / 2U;

  const uint16_t start = (mid > half) ? (uint16_t)(mid - half) : 0U;
  const uint16_t end   = (start + middle_cols < res->cols)
                         ? start + middle_cols
                         : res->cols;
  const uint16_t total = end - start;

  uint16_t open_cols = 0U;
  for (uint16_t c = start; c < end; c++) {
    if (res->horizon[c] > 1U) { open_cols++; }
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

  const uint16_t rows_minus1 = (uint16_t)(res->rows - 1U);

  for (uint16_t c = 0U; c < res->cols; c++) {
    const uint16_t h = res->horizon[c];
    if (h == 0U) { continue; }

    total += h;
    sum_x += (uint32_t)c * h;

    const uint16_t top_row = (uint16_t)(res->rows - h);
    const uint16_t avg_row = (uint16_t)((top_row + rows_minus1) / 2U);
    sum_y += (uint32_t)avg_row * h;
  }

  res->ground_count = total;

  if (total > 0U) {
    res->x_c = (int32_t)roundf((float)sum_x / (float)total - (res->cols * 0.5f));
    res->y_c = (int32_t)roundf((res->rows * 0.5f) - (float)sum_y / (float)total);
  } else {
    res->x_c = 0;
    res->y_c = 0;
  }
}

/* -------------------------------------------------------------------------- */
/* Finding Carpets Core Logic                                                 */
/* -------------------------------------------------------------------------- */

#define CARPET_EDGE_THRESHOLD 150U
uint8_t dilation_iterations    = 2;
uint8_t correction_iterations  = 3;
uint8_t erode_again_iterations = 5;

static void dilate_obstacles(uint16_t cols, uint16_t rows, uint8_t iterations)
{
  for (uint8_t iter = 0U; iter < iterations; iter++) {
    memcpy(ground_temp, ground_small, (size_t)rows * GS_MAX_COLS);

    for (uint16_t r = 0U; r < rows; r++) {
      const uint8_t *src_row = ground_temp[r];
      for (uint16_t c = 0U; c < cols; c++) {
        if (src_row[c] == 0U) {
          if (c > 0U)         { ground_small[r][c - 1U] = 0U; }
          if (c < cols - 1U)  { ground_small[r][c + 1U] = 0U; }
          if (r > 0U)         { ground_small[r - 1U][c] = 0U; }
          if (r < rows - 1U)  { ground_small[r + 1U][c] = 0U; }
          if (r > 0U && c > 0U)           { ground_small[r - 1U][c - 1U] = 0U; }
          if (r > 0U && c < cols - 1U)    { ground_small[r - 1U][c + 1U] = 0U; }
          if (r < rows - 1U && c > 0U)    { ground_small[r + 1U][c - 1U] = 0U; }
          if (r < rows - 1U && c < cols - 1U) { ground_small[r + 1U][c + 1U] = 0U; }
        }
      }
    }
  }
}

static void correct_carpets_generic(uint16_t cols, uint16_t rows,
                                    uint8_t iterations,
                                    uint8_t eval_val,
                                    uint8_t count_val,
                                    uint8_t flip_val)
{
  const uint8_t THRESHOLD = 5U;

  for (uint8_t iter = 0U; iter < iterations; iter++) {
    memcpy(carpet_temp, carpet_small, (size_t)rows * GS_MAX_COLS);

    for (uint16_t r = 0U; r < rows; r++) {
      const uint8_t *src_row = carpet_temp[r];
      for (uint16_t c = 0U; c < cols; c++) {
        if (src_row[c] != eval_val) { continue; }

        uint8_t count = 0U;
        if (c > 0U          && carpet_temp[r][c - 1U]          == count_val) { count++; }
        if (c < cols - 1U   && carpet_temp[r][c + 1U]          == count_val) { count++; }
        if (r > 0U          && carpet_temp[r - 1U][c]          == count_val) { count++; }
        if (r < rows - 1U   && carpet_temp[r + 1U][c]          == count_val) { count++; }
        if (r > 0U && c > 0U          && carpet_temp[r - 1U][c - 1U] == count_val) { count++; }
        if (r > 0U && c < cols - 1U   && carpet_temp[r - 1U][c + 1U] == count_val) { count++; }
        if (r < rows - 1U && c > 0U   && carpet_temp[r + 1U][c - 1U] == count_val) { count++; }
        if (r < rows - 1U && c < cols - 1U && carpet_temp[r + 1U][c + 1U] == count_val) { count++; }

        if (count >= THRESHOLD) {
          carpet_small[r][c] = flip_val;
        }
      }
    }
  }
}

static void find_carpet(struct image_t *img, uint16_t cols, uint16_t rows)
{
  memset(carpet_small, 0, sizeof(carpet_small));

  const uint16_t row_start = (uint16_t)((2U * rows) / 3U);
  const uint16_t max_x = logical_width(img);
  const uint16_t max_y = logical_height(img);
  const uint8_t  *buf  = (const uint8_t *)img->buf;
  const uint16_t  img_w = img->w;

  for (uint16_t r = row_start; r < rows; r++) {
    for (uint16_t c = 0U; c < cols; c++) {
      const uint16_t x0 = (uint16_t)(c * ground_downsize_x);
      const uint16_t y0 = (uint16_t)(r * ground_downsize_y);

      if (x0 >= max_x || y0 >= max_y) { continue; }

      uint32_t total_edge_magnitude = 0U;

      const uint16_t x_end = (uint16_t)(x0 + ground_downsize_x);
      const uint16_t y_end = (uint16_t)(y0 + ground_downsize_y);
      const uint16_t xe = (x_end < max_x) ? x_end : max_x;
      const uint16_t ye = (y_end < max_y) ? y_end : max_y;

      for (uint16_t y = y0; y < ye; y++) {
        for (uint16_t x = x0; x < xe; x++) {
          const uint16_t lw1 = max_x - 1U;
          const uint16_t lh1 = max_y - 1U;

          /* left neighbour (x-1, y) */
          const uint16_t xl = (x > 0U) ? (x - 1U) : 0U;
          uint16_t sx, sy;
          sx = (uint16_t)(img_w - 1U - (xl <= lh1 ? y : lh1));
          sy = (xl <= lw1) ? xl : lw1;
          uint8_t Y_left, Ud, Vd;
          get_yuv422_pixel_fast(buf, img_w, sx, sy, &Y_left, &Ud, &Vd);

          /* right neighbour (x+1, y) */
          const uint16_t xr2 = (x < lw1) ? (x + 1U) : lw1;
          sx = (uint16_t)(img_w - 1U - (y <= lh1 ? y : lh1));
          sy = (xr2 <= lw1) ? xr2 : lw1;
          uint8_t Y_right;
          get_yuv422_pixel_fast(buf, img_w, sx, sy, &Y_right, &Ud, &Vd);

          /* up neighbour (x, y-1) */
          const uint16_t yu2 = (y > 0U) ? (y - 1U) : 0U;
          sx = (uint16_t)(img_w - 1U - (yu2 <= lh1 ? yu2 : lh1));
          sy = (x <= lw1) ? x : lw1;
          uint8_t Y_up;
          get_yuv422_pixel_fast(buf, img_w, sx, sy, &Y_up, &Ud, &Vd);

          /* down neighbour (x, y+1) */
          const uint16_t yd2 = (y < lh1) ? (y + 1U) : lh1;
          sx = (uint16_t)(img_w - 1U - (yd2 <= lh1 ? yd2 : lh1));
          sy = (x <= lw1) ? x : lw1;
          uint8_t Y_down;
          get_yuv422_pixel_fast(buf, img_w, sx, sy, &Y_down, &Ud, &Vd);

          const int16_t Gx = (int16_t)Y_right - (int16_t)Y_left;
          const int16_t Gy = (int16_t)Y_down  - (int16_t)Y_up;
          total_edge_magnitude += (uint32_t)(abs(Gx) + abs(Gy));
        }
      }

      if (total_edge_magnitude > CARPET_EDGE_THRESHOLD) {
        carpet_small[r][c] = 1U;
      }
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

  if (img == NULL || img->buf == NULL) { return 0U; }
  if (ground_downsize_x == 0U || ground_downsize_y == 0U) { return 0U; }

  const uint16_t lw = logical_width(img);
  const uint16_t lh = logical_height(img);

  uint16_t cols = lw / ground_downsize_x;
  uint16_t rows = lh / ground_downsize_y;

  if (cols == 0U || rows == 0U) { return 0U; }
  if (cols > GS_MAX_COLS) { cols = GS_MAX_COLS; }
  if (rows > GS_MAX_ROWS) { rows = GS_MAX_ROWS; }

  res->cols = cols;
  res->rows = rows;

  find_carpet(img, cols, rows);
  correct_carpets_generic(cols, rows, correction_iterations,  0U, 1U, 1U);
  correct_carpets_generic(cols, rows, erode_again_iterations, 1U, 0U, 0U);

  build_ground_map(img, cols, rows);

  /* Merge maps — fused into a single pass over the grid */
  for (uint16_t r = 0U; r < rows; r++) {
    for (uint16_t c = 0U; c < cols; c++) {
      ground_small[r][c] = (ground_small[r][c] | carpet_small[r][c])
                           & (uint8_t)(~tree_small[r][c]);
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

  if (!updated) { return; }
}

void ground_seg_get_result(struct ground_seg_result_t *out)
{
  pthread_mutex_lock(&ground_seg_mutex);
  memcpy(out, &ground_seg_shared.result, sizeof(*out));
  pthread_mutex_unlock(&ground_seg_mutex);
}