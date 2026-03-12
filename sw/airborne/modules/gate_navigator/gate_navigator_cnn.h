/*
 * gate_navigator_cnn.h
 * --------------------
 * CNN-based gate navigation module.
 * Reads heading + confidence from gate_cnn_result and flies
 * through the gate in GUIDED mode.
 *
 * Completely separate from gate_navigator — do NOT load both
 * in the same airframe.
 *
 * Place in: sw/airborne/modules/gate_navigator/gate_navigator_cnn.h
 */

#ifndef GATE_NAVIGATOR_CNN_H
#define GATE_NAVIGATOR_CNN_H

extern void gate_navigator_cnn_init(void);
extern void gate_navigator_cnn_periodic(void);

/* Tunable parameters — all adjustable from the GCS Settings panel */
extern float cnn_nav_forward_speed;    /* m/s forward speed during approach/pass  */
extern float cnn_nav_conf_thresh;      /* min CNN confidence to consider gate found */
extern float cnn_nav_align_thresh;     /* |heading| below this = aligned           */
extern float cnn_nav_search_rate;      /* deg/s yaw rate while searching            */
extern float cnn_nav_yaw_gain;         /* deg/s per unit heading error              */
extern float cnn_nav_approach_time_s;  /* seconds forward flight before PASS        */
extern float cnn_nav_pass_time_s;      /* seconds to fly through the gate           */

#endif /* GATE_NAVIGATOR_CNN_H */