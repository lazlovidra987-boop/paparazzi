/*
 * Ground segmentation module for Paparazzi
 *
 * Purpose:
 * - Read frames from one camera in YUV422 format
 * - Detect ground pixels using Y, Cb, Cr thresholds
 * - Compute centroid of detected ground region
 * - Split detected pixels into:
 *     1) vertical regions: left / center / right
 *     2) horizontal regions: top / middle / bottom
 * - Optionally draw detected pixels and region divider lines in the image
 * - Send result through ABI so other modules can use it
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

/* Debug print macro */
#define GS_PRINT(string, ...) fprintf(stderr, "[cv_ground_seg->%s()] " string, __FUNCTION__, ##__VA_ARGS__)

/*
 * If not defined in XML, run at camera frame rate.
 */
#ifndef GROUND_SEGMENTATION_FPS
#define GROUND_SEGMENTATION_FPS 0
#endif

/* Mutex for sharing detection result safely between callback and periodic task */
static pthread_mutex_t ground_seg_mutex;

/*
 * Ground segmentation thresholds in YUV / YCbCr space.
 * These can be tuned from Paparazzi settings.
 */
uint8_t ground_lum_min = 50;
uint8_t ground_lum_max = 150;
uint8_t ground_cb_min  = 70;
uint8_t ground_cb_max  = 120;
uint8_t ground_cr_min  = 50;
uint8_t ground_cr_max  = 140;

/* Whether detected pixels should be highlighted in the image */
bool ground_draw = true;



/* Shared result storage */
static struct ground_seg_result_t ground_seg_result;

/*
 * Forward declaration
 */
static uint32_t ground_seg_analyse_image(struct image_t *img,
                                         int32_t *p_xc,
                                         int32_t *p_yc,
                                         uint32_t *p_left_count,
                                         uint32_t *p_center_count,
                                         uint32_t *p_right_count,
                                         uint32_t *p_top_count,
                                         uint32_t *p_middle_count,
                                         uint32_t *p_bottom_count,
                                         bool draw);

/*
 * Main image-processing callback.
 * Called automatically whenever a new camera frame is available.
 */
static struct image_t *ground_seg_process_image(struct image_t *img, uint8_t camera_id)
{
  (void)camera_id; /* single-camera module */

  int32_t x_c = 0;
  int32_t y_c = 0;

  uint32_t left_count = 0;
  uint32_t center_count = 0;
  uint32_t right_count = 0;

  uint32_t top_count = 0;
  uint32_t middle_count = 0;
  uint32_t bottom_count = 0;

  uint32_t count = ground_seg_analyse_image(img,
                                            &x_c,
                                            &y_c,
                                            &left_count,
                                            &center_count,
                                            &right_count,
                                            &top_count,
                                            &middle_count,
                                            &bottom_count,
                                            ground_draw);

  pthread_mutex_lock(&ground_seg_mutex);
  ground_seg_result.x_c = x_c;
  ground_seg_result.y_c = y_c;
  ground_seg_result.pixel_count = count;

  ground_seg_result.left_count = left_count;
  ground_seg_result.center_count = center_count;
  ground_seg_result.right_count = right_count;

  ground_seg_result.top_count = top_count;
  ground_seg_result.middle_count = middle_count;
  ground_seg_result.bottom_count = bottom_count;

  ground_seg_result.updated = true;
  pthread_mutex_unlock(&ground_seg_mutex);

  GS_PRINT("Frame processed: total=%u | L=%u C=%u R=%u | T=%u M=%u B=%u | x_c=%d y_c=%d\n",
           count,
           left_count, center_count, right_count,
           top_count, middle_count, bottom_count,
           x_c, y_c);

  return img;
}

/*
 * Module initialization.
 */
void ground_segmentation_init(void)
{
  memset(&ground_seg_result, 0, sizeof(ground_seg_result));
  pthread_mutex_init(&ground_seg_mutex, NULL);

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

/*
 * Analyse the image:
 * - classify pixels as ground / not ground
 * - compute centroid
 * - count per vertical and horizontal region
 * - optionally draw segmentation and divider lines
 *
 * Image format is YUV422:
 * each pair of pixels is stored as U Y1 V Y2
 */
static uint32_t ground_seg_analyse_image(struct image_t *img,
                                         int32_t *p_xc,
                                         int32_t *p_yc,
                                         uint32_t *p_left_count,
                                         uint32_t *p_center_count,
                                         uint32_t *p_right_count,
                                         uint32_t *p_top_count,
                                         uint32_t *p_middle_count,
                                         uint32_t *p_bottom_count,
                                         bool draw)
{
  uint32_t cnt = 0;
  uint32_t tot_x = 0;
  uint32_t tot_y = 0;

  uint32_t left_count = 0;
  uint32_t center_count = 0;
  uint32_t right_count = 0;

  uint32_t top_count = 0;
  uint32_t middle_count = 0;
  uint32_t bottom_count = 0;

  uint8_t *buffer = img->buf;

  const uint16_t x_div1 = img->w / 3;
  const uint16_t x_div2 = (2 * img->w) / 3;
  const uint16_t y_div1 = img->h / 3;
  const uint16_t y_div2 = (2 * img->h) / 3;

  for (uint16_t y = 0; y < img->h; y++) {
    for (uint16_t x = 0; x < img->w; x++) {

      uint8_t *yp;
      uint8_t *up;
      uint8_t *vp;

      /* Recover Y, U, V pointers depending on even/odd x in YUV422 */
      if ((x % 2) == 0) {
        up = &buffer[y * 2 * img->w + 2 * x];
        yp = &buffer[y * 2 * img->w + 2 * x + 1];
        vp = &buffer[y * 2 * img->w + 2 * x + 2];
      } else {
        up = &buffer[y * 2 * img->w + 2 * x - 2];
        vp = &buffer[y * 2 * img->w + 2 * x];
        yp = &buffer[y * 2 * img->w + 2 * x + 1];
      }

      /*
       * Draw divider lines for debugging:
       * - two vertical lines split left/center/right
       * - two horizontal lines split top/middle/bottom
       */
      if (draw) {
        if (x == x_div1 || x == x_div2 || y == y_div1 || y == y_div2) {
          *yp = 255;  /* bright line */
        }
      }

      /*
       * Check if pixel lies inside the ground threshold box.
       */
      if ((*yp >= ground_lum_min) && (*yp <= ground_lum_max) &&
          (*up >= ground_cb_min)  && (*up <= ground_cb_max)  &&
          (*vp >= ground_cr_min)  && (*vp <= ground_cr_max)) {

        cnt++;
        tot_x += x;
        tot_y += y;

        /* Vertical split: left / center / right */
        if (x < x_div1) {
          left_count++;
        } else if (x < x_div2) {
          center_count++;
        } else {
          right_count++;
        }

        /* Horizontal split: top / middle / bottom */
        if (y < y_div1) {
          top_count++;
        } else if (y < y_div2) {
          middle_count++;
        } else {
          bottom_count++;
        }

        /*
         * Draw detected ground in a strong artificial color.
         * This is for debugging in RTP.
         */
        if (draw) {
          *yp = 150;
          *up = 40;
          *vp = 20;
        }
      }
    }
  }

  /*
   * Compute centroid relative to image center.
   * x_c > 0 : centroid right of center
   * y_c > 0 : centroid above center
   */
  if (cnt > 0) {
    *p_xc = (int32_t)roundf((tot_x / (float)cnt) - img->w * 0.5f);
    *p_yc = (int32_t)roundf(img->h * 0.5f - (tot_y / (float)cnt));
  } else {
    *p_xc = 0;
    *p_yc = 0;
  }

  *p_left_count = left_count;
  *p_center_count = center_count;
  *p_right_count = right_count;

  *p_top_count = top_count;
  *p_middle_count = middle_count;
  *p_bottom_count = bottom_count;

  return cnt;
}

/*
 * Periodic function called by Paparazzi.
 * Sends the newest segmentation result through ABI.
 *
 * For now we keep using VISUAL_DETECTION for compatibility and debugging.
 */
void ground_segmentation_periodic(void)
{
  struct ground_seg_result_t local_result;

  pthread_mutex_lock(&ground_seg_mutex);
  memcpy(&local_result, &ground_seg_result, sizeof(local_result));
  ground_seg_result.updated = false;
  pthread_mutex_unlock(&ground_seg_mutex);

  if (local_result.updated) {
    AbiSendMsgVISUAL_DETECTION(
      COLOR_OBJECT_DETECTION1_ID,
      local_result.x_c,
      local_result.y_c,
      0,
      0,
      local_result.pixel_count,
      0
    );

    GS_PRINT("Ground detected: total=%u | L=%u C=%u R=%u | T=%u M=%u B=%u | x_c=%d y_c=%d\n",
             local_result.pixel_count,
             local_result.left_count,
             local_result.center_count,
             local_result.right_count,
             local_result.top_count,
             local_result.middle_count,
             local_result.bottom_count,
             local_result.x_c,
             local_result.y_c);
  }
}



void ground_seg_get_result(struct ground_seg_result_t *out)
{
  pthread_mutex_lock(&ground_seg_mutex);
  memcpy(out, &ground_seg_result, sizeof(*out));
  pthread_mutex_unlock(&ground_seg_mutex);
}