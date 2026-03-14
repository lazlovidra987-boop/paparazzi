/*
 * gate_cnn.h
 * ----------
 * Paparazzi module header for the CNN-based gate detector.
 *
 * This module reads the front camera, runs DirectionCNN on each frame,
 * and publishes the result via ABI as a VISUAL_DETECTION message.
 *
 * Results can be read by a guidance module using:
 *   AbiBindMsgVISUAL_DETECTION(GATE_CNN_ABI_ID, &ev, callback);
 */

#ifndef GATE_CNN_H
#define GATE_CNN_H

/* ABI sender ID — must be unique across all modules in your airframe.
 * Check modules/core/abi_sender_ids.h for used IDs and pick a free one. */
#ifndef GATE_CNN_ABI_ID
#define GATE_CNN_ABI_ID 255
#endif

/* Camera to attach to. Override in airframe XML with:
 *   <define name="GATE_CNN_CAMERA" value="front_camera"/> */
#ifndef GATE_CNN_CAMERA
#define GATE_CNN_CAMERA front_camera
#endif

/* Frames per second passed to cv_add_to_device.
 * 0 = run at camera rate. 10 is safe for the Bebop. */
#ifndef GATE_CNN_FPS
#define GATE_CNN_FPS 10
#endif

/* Confidence threshold: gate is considered detected above this value. */
#ifndef GATE_CNN_CONF_THRESHOLD
#define GATE_CNN_CONF_THRESHOLD 0.5f
#endif

/* Public result struct — read this from your guidance module if needed. */
struct gate_cnn_result_t {
  float   heading;      /* normalised heading [-1, 1] */
  float   gate_measure;   /* gate visible probability [0, 1] */
  uint8_t has_gate;     /* 1 if confidence > threshold, else 0 */
  uint32_t frame_count; /* number of frames processed */
};

extern struct gate_cnn_result_t gate_cnn_result;
extern float gate_cnn_conf_threshold;

/* Paparazzi module interface */
void gate_cnn_init(void);
void gate_cnn_periodic(void);

#endif /* GATE_CNN_H */