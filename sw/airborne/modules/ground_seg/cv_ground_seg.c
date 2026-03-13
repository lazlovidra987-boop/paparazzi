/*
 * Ground segmentation module for Paparazzi
 *
 * Purpose:
 * - Read frames from one camera in YUV422 format
 * - Detect ground pixels using Y, Cb, Cr thresholds
 * - Compute centroid of detected ground region
 * - Optionally draw the detected pixels in the image
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
#include <pthread.h>

/* Simple debug print macro */
#define GS_PRINT(string, ...) fprintf(stderr, "[cv_ground_seg->%s()] " string, __FUNCTION__, ##__VA_ARGS__)

/*
 * If the XML does not define a custom FPS,
 * use 0 = run at camera frame rate.
 */
#ifndef GROUND_SEGMENTATION_FPS
#define GROUND_SEGMENTATION_FPS 0
#endif

/* Mutex protects shared segmentation result */
static pthread_mutex_t ground_seg_mutex;

/*
 * Threshold values for ground detection in YCbCr / YUV space.
 * These can be tuned from the Paparazzi settings interface.
 */
uint8_t ground_lum_min = 50;
uint8_t ground_lum_max = 150;
uint8_t ground_cb_min  = 70;
uint8_t ground_cb_max  = 120;
uint8_t ground_cr_min  = 50;
uint8_t ground_cr_max  = 140;

/* Whether detected ground pixels should be highlighted in the image */
bool ground_draw = true;

/*
 * Struct holding the latest ground segmentation result.
 *
 * x_c, y_c:
 *   centroid coordinates relative to image center
 *
 * pixel_count:
 *   number of pixels classified as ground
 *
 * updated:
 *   true when a new result is available
 */
struct ground_seg_result_t {
  int32_t x_c;
  int32_t y_c;
  uint32_t pixel_count;
  bool updated;
};

/* Shared result storage */
static struct ground_seg_result_t ground_seg_result;

/*
 * Forward declaration:
 * scan image, segment ground pixels, compute centroid
 */
static uint32_t ground_seg_find_centroid(struct image_t *img,
                                         int32_t *p_xc,
                                         int32_t *p_yc,
                                         bool draw);

/*
 * Main image-processing callback.
 * This is called automatically whenever a new camera frame is available.
 */
static struct image_t *ground_seg_process_image(struct image_t *img, uint8_t camera_id)
{
  /* We do not use the camera_id because this module uses only one camera */
  (void)camera_id;

  int32_t x_c = 0;
  int32_t y_c = 0;

  /* Run segmentation and compute centroid */
  uint32_t count = ground_seg_find_centroid(img, &x_c, &y_c, ground_draw);
  GS_PRINT("Frame processed: count=%u x_c=%d y_c=%d\n", count, x_c, y_c);


  /* Store the result safely for the periodic task */
  pthread_mutex_lock(&ground_seg_mutex);
  ground_seg_result.x_c = x_c;
  ground_seg_result.y_c = y_c;
  ground_seg_result.pixel_count = count;
  ground_seg_result.updated = true;
  pthread_mutex_unlock(&ground_seg_mutex);

  return img;
}

/*
 * Module initialization.
 * Called once at startup.
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
 * Find centroid of all pixels that satisfy the ground thresholds.
 *
 * Input image format: YUV422
 *
 * In YUV422, every pair of pixels is stored as:
 *   U Y1 V Y2
 *
 * So for even and odd x positions, the indexing differs.
 */
static uint32_t ground_seg_find_centroid(struct image_t *img,
                                         int32_t *p_xc,
                                         int32_t *p_yc,
                                         bool draw)
{
  uint32_t cnt = 0;
  uint32_t tot_x = 0;
  uint32_t tot_y = 0;

  uint8_t *buffer = img->buf;

  /* Loop over all image pixels */
  for (uint16_t y = 0; y < img->h; y++) {
    for (uint16_t x = 0; x < img->w; x++) {

      uint8_t *yp;
      uint8_t *up;
      uint8_t *vp;

      /*
       * Reconstruct Y, U, V pointers depending on whether x is even or odd.
       */
      if ((x % 2) == 0) {
        /* Even pixel: U Y1 V Y2 */
        up = &buffer[y * 2 * img->w + 2 * x];
        yp = &buffer[y * 2 * img->w + 2 * x + 1];
        vp = &buffer[y * 2 * img->w + 2 * x + 2];
      } else {
        /* Odd pixel uses same U and V but its own Y */
        up = &buffer[y * 2 * img->w + 2 * x - 2];
        vp = &buffer[y * 2 * img->w + 2 * x];
        yp = &buffer[y * 2 * img->w + 2 * x + 1];
      }

      /*
       * Check if pixel lies inside the ground threshold box in YUV space.
       */
      if ((*yp >= ground_lum_min) && (*yp <= ground_lum_max) &&
          (*up >= ground_cb_min)  && (*up <= ground_cb_max)  &&
          (*vp >= ground_cr_min)  && (*vp <= ground_cr_max)) {

        cnt++;
        tot_x += x;
        tot_y += y;

        if (draw) {
          /* very visible debug color */
          *yp = 150;
          *up = 40;
          *vp = 20;
        }
      }
    }
  }

  /*
   * If any ground pixels were found, compute centroid relative to image center.
   */
  if (cnt > 0) {
    *p_xc = (int32_t)roundf((tot_x / (float)cnt) - img->w * 0.5f);
    *p_yc = (int32_t)roundf(img->h * 0.5f - (tot_y / (float)cnt));
  } else {
    *p_xc = 0;
    *p_yc = 0;
  }

  return cnt;
}


/*
 * Periodic function called by Paparazzi.
 * Sends the newest segmentation result through ABI.
 */
void ground_segmentation_periodic(void)
{
  struct ground_seg_result_t local_result;

  /* Copy result safely */
  pthread_mutex_lock(&ground_seg_mutex);
  memcpy(&local_result, &ground_seg_result, sizeof(local_result));
  ground_seg_result.updated = false;
  pthread_mutex_unlock(&ground_seg_mutex);

  if (local_result.updated) {
    /*
     * Send centroid and pixel count as a visual detection message.
     * You can later replace COLOR_OBJECT_DETECTION1_ID with your own ID if needed.
     */
    AbiSendMsgVISUAL_DETECTION(
      COLOR_OBJECT_DETECTION1_ID,
      local_result.x_c,
      local_result.y_c,
      0,
      0,
      local_result.pixel_count,
      0
    );

    GS_PRINT("Ground detected: x_c=%d y_c=%d count=%u\n",
             local_result.x_c,
             local_result.y_c,
             local_result.pixel_count);
  }
}



// Each frame:

// the callback reads the image

// every pixel is converted from raw YUV422 memory layout into Y, U, V

// the code checks whether that pixel matches the ground thresholds

// if yes:

// pixel count increases

// x and y are added for centroid

// pixel is brightened if drawing is enabled

// Then:

// the centroid is computed

// the result is stored - the periodic task sends it via ABI