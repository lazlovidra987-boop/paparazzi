#ifndef CV_GROUND_SEG_H
#define CV_GROUND_SEG_H

#include <stdint.h>
#include <stdbool.h>
#include "modules/computer_vision/cv.h"

/* Ground threshold settings */
extern uint8_t ground_lum_min;
extern uint8_t ground_lum_max;
extern uint8_t ground_cb_min;
extern uint8_t ground_cb_max;
extern uint8_t ground_cr_min;
extern uint8_t ground_cr_max;

/* Draw detected ground pixels on the image */
extern bool ground_draw;

/* Module functions */
extern void ground_segmentation_init(void);
extern void ground_segmentation_periodic(void);

#endif /* CV_GROUND_SEG_H */
/*
 * Simple ground segmentation module using color thresholding in YUV space.
 *
 * This module processes images from a specified camera, identifies pixels that
 * fall within defined YUV thresholds for the ground, and computes the centroid
 * of those pixels relative to the image center. The results are sent through
 * ABI as visual detections.
 *
 * The code is structured as follows:
 * - Initialization function to set up the module and register the image processing callback.
 * - Image processing function that applies the color thresholding and computes the centroid.
 * - Periodic function that sends the latest segmentation results through ABI.
 *
 * Note: This is a simple example for demonstration purposes. For more robust
 * ground segmentation, consider using more advanced techniques such as machine
 * learning-based methods or additional sensor data.
 */