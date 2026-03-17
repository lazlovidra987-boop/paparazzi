/*
 * ground_seg_nav.h
 *
 * Navigation module using ground segmentation for obstacle avoidance.
 * Reads the 3x3 grid output from cv_ground_seg and steers the drone
 * toward the safest direction using GUIDED mode.
 */

#ifndef GROUND_SEG_NAV_H
#define GROUND_SEG_NAV_H

/* Tunable settings (exposed to GCS) */
extern float gsn_max_speed;      // max forward speed [m/s]
extern float gsn_heading_rate;   // turning rate [rad/s]
extern float gsn_floor_frac;     // min floor fraction before OUT_OF_BOUNDS
extern float gsn_obstacle_frac;  // min center ground fraction before OBSTACLE_FOUND

extern void ground_seg_nav_init(void);
extern void ground_seg_nav_periodic(void);

extern void orange_avoider_guided_retreat(void);

#endif /* GROUND_SEG_NAV_H */