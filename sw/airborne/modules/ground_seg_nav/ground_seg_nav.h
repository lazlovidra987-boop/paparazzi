#ifndef GROUND_SEG_NAV_H
#define GROUND_SEG_NAV_H

/*
 * ground_seg_nav.h
 *
 * Navigation module using horizon-based ground segmentation from cv_ground_seg.
 *
 * The module reads horizon-based free-ground estimates and decides:
 * - move forward when the path ahead looks clear
 * - stop and start a turn when blocked
 * - keep turning until the center region is open again
 *
 * Note:
 * Legacy Paparazzi setting names are kept for XML/settings compatibility.
 * Their original names are misleading, so see the comments below carefully.
 */

/* Maximum forward body speed [m/s] */
extern float gsn_max_speed;

/* Heading/yaw rate while searching for a free direction [rad/s] */
extern float gsn_heading_rate;

/*
 * Legacy names kept for compatibility.
 *
 * New meaning:
 * gsn_floor_frac
 *   Minimum mean visible ground ahead required to keep moving forward.
 *   In practice this is compared against the average center horizon.
 *
 * gsn_obstacle_frac
 *   Minimum mean visible ground ahead required to leave turning mode
 *   and continue forward.
 *
 * Important:
 * The names are legacy only. They do NOT literally mean "floor fraction"
 * and "obstacle fraction" anymore.
 */
extern float gsn_floor_frac;
extern float gsn_obstacle_frac;

/* Module init */
extern void ground_seg_nav_init(void);

/* Periodic navigation update */
extern void ground_seg_nav_periodic(void);

/*
 * Required by guided flight plan / compatibility with existing setup.
 * Actual retreat behaviour can still be handled by the nav state machine.
 */
extern void orange_avoider_guided_retreat(void);

#endif /* GROUND_SEG_NAV_H */