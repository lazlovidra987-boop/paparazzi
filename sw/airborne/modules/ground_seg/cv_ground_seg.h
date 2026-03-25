#ifndef CV_GROUND_SEG_H
#define CV_GROUND_SEG_H

#include <stdint.h>
#include <stdbool.h>
#include "modules/computer_vision/cv.h"

/* Maximum size of the downsized segmentation grid */
#define GS_MAX_COLS 160
#define GS_MAX_ROWS 120

/* YUV threshold settings for ground classification */
extern uint8_t ground_lum_min;
extern uint8_t ground_lum_max;
extern uint8_t ground_cb_min;
extern uint8_t ground_cb_max;
extern uint8_t ground_cr_min;
extern uint8_t ground_cr_max;

extern uint8_t tree_lum_min;
extern uint8_t tree_lum_max;
extern uint8_t tree_cb_min;
extern uint8_t tree_cb_max;
extern uint8_t tree_cr_min;
extern uint8_t tree_cr_max;

/*
 * Segmentation settings
 *
 * ground_downsize_x / ground_downsize_y
 *   Block size used for downsizing the image.
 *
 * ground_min_black
 *   Minimum consecutive black cells in a bottom-up column scan that are
 *   accepted as a real obstacle boundary. Shorter black gaps are filled back
 *   in as ground.
 *
 * ground_middle_cols
 *   Number of center columns used for obstacle detection.
 */
extern uint8_t ground_downsize_x;
extern uint8_t ground_downsize_y;
extern uint8_t ground_min_black;
extern uint8_t ground_middle_cols;

/* Debug drawing on the camera image */
extern bool ground_draw;

/*
 * Ground segmentation result.
 *
 * horizon[c]:
 *   0  = no visible ground detected in column c
 *   >0 = visible ground depth from image bottom upward
 *
 * The horizon is built from the downsized binary ground map using
 * a bottom-up scan with short-gap reclassification.
 *
 * Scores:
 *   Sum of horizon values over left / center / right image regions.
 *   Higher score means more visible free ground in that region.
 */
struct ground_seg_result_t {
  /* Actual downsized grid size used for the current image */
  uint16_t cols;
  uint16_t rows;

  /* Visible ground depth per downsized column */
  uint16_t horizon[GS_MAX_COLS];

  /* Region scores derived from horizon[] */
  uint32_t left_score;
  uint32_t center_score;
  uint32_t right_score;

  /* Total visible ground amount reconstructed from horizon[] */
  uint32_t ground_count;

  /*
   * Centroid of reconstructed visible ground,
   * expressed relative to the downsized image center.
   *
   * x_c > 0 : centroid is to the right
   * x_c < 0 : centroid is to the left
   * y_c > 0 : centroid is above center
   * y_c < 0 : centroid is below center
   */
  int32_t x_c;
  int32_t y_c;

  /* True if the middle viewing region is considered blocked */
  bool obstacle_ahead;
};

/* Module interface */
extern void ground_segmentation_init(void);
extern void ground_segmentation_periodic(void);
extern void ground_seg_get_result(struct ground_seg_result_t *out);

#endif /* CV_GROUND_SEG_H */