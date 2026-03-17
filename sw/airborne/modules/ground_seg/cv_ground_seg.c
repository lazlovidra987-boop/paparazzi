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
 * - No edge detection here
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

uint8_t ground_downsize_x  = 4;
uint8_t ground_downsize_y  = 4;
uint8_t ground_min_black   = 5;
uint8_t ground_middle_cols = 10;

bool ground_draw = true;

/* Downsized binary classification map in LOGICAL rotated coordinates */
static uint8_t ground_small[GS_MAX_ROWS][GS_MAX_COLS];

/* -------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* -------------------------------------------------------------------------- */

static inline bool should_print_debug(void)
{
  return (gs_frame_counter % GS_DEBUG_EVERY_N_FRAMES) == 0U;
}

static void reset_result(struct ground_seg_result_t *res)
{
  memset(res, 0, sizeof(*res));
}

static inline uint16_t clamp_u16(uint16_t v, uint16_t lo, uint16_t hi)
{
  if (v < lo) {
    return lo;
  }
  if (v > hi) {
    return hi;
  }
  return v;
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

/*
 * Draw on the PHYSICAL image buffer, but using LOGICAL rotated coordinates.
 */
static void draw_classified_pixel(struct image_t *img, uint16_t x, uint16_t y, bool is_ground)
{
  if (!ground_draw || img == NULL || img->buf == NULL || img->w < 2U || img->h == 0U) {
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

  uint8_t Uv = is_ground ? 40U  : 128U;
  uint8_t Yv = is_ground ? 150U : 60U;
  uint8_t Vv = is_ground ? 20U  : 128U;

  buffer[base]     = Uv;
  buffer[base + 2] = Vv;

  if ((src_x & 1U) == 0U) {
    buffer[base + 1U] = Yv;
  } else {
    buffer[base + 3U] = Yv;
  }
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

/* -------------------------------------------------------------------------- */
/* Debug                                                                      */
/* -------------------------------------------------------------------------- */

static void debug_print_image_info(struct image_t *img, uint16_t cols, uint16_t rows)
{
  if (!should_print_debug()) {
    return;
  }

  GS_PRINT("phys_w=%u phys_h=%u | logical_w=%u logical_h=%u | down_x=%u down_y=%u | cols=%u rows=%u\n",
           img->w, img->h,
           logical_width(img), logical_height(img),
           ground_downsize_x, ground_downsize_y,
           cols, rows);
}

static void debug_print_raw_samples(struct image_t *img)
{
  if (!should_print_debug()) {
    return;
  }

  uint16_t lw = logical_width(img);
  uint16_t lh = logical_height(img);

  uint16_t y_test  = lh / 2U;
  uint16_t x_left  = lw / 6U;
  uint16_t x_mid   = lw / 2U;
  uint16_t x_right = (uint16_t)((5U * lw) / 6U);

  uint8_t Y, U, V;

  get_yuv422_pixel(img, x_left, y_test, &Y, &U, &V);
  GS_PRINT("RAW LEFT  x=%u y=%u | Y=%u U=%u V=%u | ground=%d\n",
           x_left, y_test, Y, U, V, is_ground_yuv(Y, U, V));

  get_yuv422_pixel(img, x_mid, y_test, &Y, &U, &V);
  GS_PRINT("RAW MID   x=%u y=%u | Y=%u U=%u V=%u | ground=%d\n",
           x_mid, y_test, Y, U, V, is_ground_yuv(Y, U, V));

  get_yuv422_pixel(img, x_right, y_test, &Y, &U, &V);
  GS_PRINT("RAW RIGHT x=%u y=%u | Y=%u U=%u V=%u | ground=%d\n",
           x_right, y_test, Y, U, V, is_ground_yuv(Y, U, V));
}

static void debug_print_map_counts(uint16_t cols, uint16_t rows)
{
  if (!should_print_debug()) {
    return;
  }

  uint32_t left = 0U;
  uint32_t center = 0U;
  uint32_t right = 0U;

  uint16_t c1 = cols / 3U;
  uint16_t c2 = (2U * cols) / 3U;

  for (uint16_t r = 0U; r < rows; r++) {
    for (uint16_t c = 0U; c < cols; c++) {
      if (ground_small[r][c] != 0U) {
        if (c < c1) {
          left++;
        } else if (c < c2) {
          center++;
        } else {
          right++;
        }
      }
    }
  }

  GS_PRINT("MAP | L=%u C=%u R=%u\n", left, center, right);

  if (cols >= 6U) {
    uint16_t band_w = cols / 6U;
    fprintf(stderr, "[cv_ground_seg->%s()] MAP BANDS:", __FUNCTION__);
    for (uint16_t b = 0U; b < 6U; b++) {
      uint16_t start = (uint16_t)(b * band_w);
      uint16_t end   = (b == 5U) ? cols : (uint16_t)((b + 1U) * band_w);
      uint32_t count = 0U;

      for (uint16_t r = 0U; r < rows; r++) {
        for (uint16_t c = start; c < end; c++) {
          if (ground_small[r][c] != 0U) {
            count++;
          }
        }
      }

      fprintf(stderr, " b%u=%u", b, count);
    }
    fprintf(stderr, "\n");
  }
}

static void debug_print_horizon_samples(const struct ground_seg_result_t *res)
{
  if (!should_print_debug() || res->cols == 0U) {
    return;
  }

  GS_PRINT("HORIZON samples:");
  for (uint16_t c = 0U; c < res->cols; c += 10U) {
    fprintf(stderr, " h[%u]=%u", c, res->horizon[c]);
  }

  if (((res->cols - 1U) % 10U) != 0U) {
    fprintf(stderr, " h[%u]=%u", res->cols - 1U, res->horizon[res->cols - 1U]);
  }

  fprintf(stderr, "\n");
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

      uint16_t xc = clamp_u16((uint16_t)(x0 + block_w / 2U), 0U, (uint16_t)(lw - 1U));
      uint16_t yc = clamp_u16((uint16_t)(y0 + block_h / 2U), 0U, (uint16_t)(lh - 1U));
      draw_classified_pixel(img, xc, yc, ground_small[r][c] != 0U);
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

  if (should_print_debug()) {
    GS_PRINT("OBSTACLE CHECK | start=%u end=%u open=%u total=%u obstacle=%d\n",
             start, end, open_cols, total, res->obstacle_ahead);
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

  debug_print_image_info(img, cols, rows);
  debug_print_raw_samples(img);

  build_ground_map(img, cols, rows);
  debug_print_map_counts(cols, rows);

  compute_horizon(res);
  debug_print_horizon_samples(res);

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

  gs_frame_counter++;

  /* Dump exactly one LOGICAL rotated raw frame for verification */
  static bool dumped = false;

  if (!dumped && img != NULL && img->buf != NULL) {
    uint16_t lw = logical_width(img);
    uint16_t lh = logical_height(img);

    FILE *f = fopen("/tmp/frame.pgm", "wb");
    if (f != NULL) {
      fprintf(f, "P5\n%u %u\n255\n", lw, lh);

      for (uint16_t y = 0U; y < lh; y++) {
        for (uint16_t x = 0U; x < lw; x++) {
          uint8_t Y, U, V;
          get_yuv422_pixel(img, x, y, &Y, &U, &V);
          fwrite(&Y, 1, 1, f);
        }
      }

      fclose(f);
      GS_PRINT("Saved rotated raw frame to /tmp/frame.pgm\n");
    } else {
      GS_PRINT("Failed to save raw frame\n");
    }

    dumped = true;
  }

  struct ground_seg_result_t local_result;
  ground_seg_analyse_image(img, &local_result);

  pthread_mutex_lock(&ground_seg_mutex);
  memcpy(&ground_seg_shared.result, &local_result, sizeof(local_result));
  ground_seg_shared.updated = true;
  pthread_mutex_unlock(&ground_seg_mutex);

  GS_PRINT("ground=%u | L=%u C=%u R=%u | obstacle=%d | x=%d y=%d\n",
           local_result.ground_count,
           local_result.left_score,
           local_result.center_score,
           local_result.right_score,
           local_result.obstacle_ahead,
           local_result.x_c,
           local_result.y_c);

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

  GS_PRINT("Ground segmentation initialized\n");
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

  AbiSendMsgVISUAL_DETECTION(
    COLOR_OBJECT_DETECTION1_ID,
    local_result.x_c,
    local_result.y_c,
    0,
    0,
    local_result.ground_count,
    local_result.obstacle_ahead ? 1 : 0
  );

  GS_PRINT("PERIODIC | L=%u C=%u R=%u | obstacle=%d\n",
           local_result.left_score,
           local_result.center_score,
           local_result.right_score,
           local_result.obstacle_ahead);
}

void ground_seg_get_result(struct ground_seg_result_t *out)
{
  pthread_mutex_lock(&ground_seg_mutex);
  memcpy(out, &ground_seg_shared.result, sizeof(*out));
  pthread_mutex_unlock(&ground_seg_mutex);
}