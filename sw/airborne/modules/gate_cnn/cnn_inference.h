/*
 * cnn_inference.h
 * ---------------
 * Public interface for the DirectionCNN forward pass.
 *
 * Include this header in gate_detector.c to call cnn_run().
 */

#ifndef CNN_INFERENCE_H
#define CNN_INFERENCE_H

/* Network input dimensions — must match model.py */
#define CNN_INPUT_H  120
#define CNN_INPUT_W  160

/* Maximum heading angle in degrees.
 * heading output is in [-1, 1]; multiply by this to get degrees. */
#define CNN_MAX_HEADING_DEG  45.0f

/* Confidence threshold above which a gate is considered detected. */
#define CNN_CONF_THRESHOLD   0.5f

/*
 * cnn_run
 * -------
 * Run one forward pass of DirectionCNN.
 *
 * Parameters
 * ----------
 *   image      : float[CNN_INPUT_H * CNN_INPUT_W]
 *                Grayscale image, pixels in [0, 1], row-major.
 *                Extract from YUV422: every second byte starting at index 1.
 *                Normalise: pixel_f = (float)pixel_byte / 255.0f
 *
 *   heading    : output — float in [-1, 1]
 *                Normalised horizontal direction to gate.
 *                Convert to degrees: heading_deg = *heading * CNN_MAX_HEADING_DEG
 *
 *   confidence : output — float in [0, 1]
 *                Probability that a gate is visible.
 *                Gate detected when *confidence > CNN_CONF_THRESHOLD
 */
void cnn_run(const float *image, float *heading, float *confidence);

#endif /* CNN_INFERENCE_H */