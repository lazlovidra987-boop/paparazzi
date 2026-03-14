/*
 * cnn_inference.h
 * ---------------
 * Public interface for the DirectionGateNet forward pass.
 *
 * This version supports the DroNet-style architecture with 5x5 stem
 * and SmallBlock residual-like layers.
 */

#ifndef CNN_INFERENCE_H
#define CNN_INFERENCE_H

/* Network input dimensions */
#define CNN_INPUT_H  120
#define CNN_INPUT_W  160

/* Conversion factor for heading output */
#define CNN_MAX_HEADING_DEG  45.0f

/* * Threshold logic depends on model output_mode:
 * - "probability": 0.5f is a typical threshold.
 * - "distance": threshold might represent meters (e.g., < 2.0m).
 */
#define CNN_GATE_THRESHOLD   0.5f

/**
 * cnn_run
 * -------
 * Run one forward pass of DirectionGateNet.
 *
 * @param image        Input grayscale pixels [0, 1], row-major (120x160).
 * @param heading      Output: Estimated heading in range [-1, 1].
 * @param gate_measure Output: Probability [0,1] or distance (m), 
 * depending on model training mode.
 */
void cnn_run(const float *image, float *heading, float *gate_measure);

#endif /* CNN_INFERENCE_H */