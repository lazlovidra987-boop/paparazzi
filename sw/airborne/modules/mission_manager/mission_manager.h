/*
 * mission_manager.h
 * -----------------
 * High-level mission module for:
 *  - explore flight for distance
 *  - obstacle avoidance
 *  - boundary recovery using global position
 *  - gate alignment and traversal using CNN detection
 */

#ifndef MISSION_MANAGER_H
#define MISSION_MANAGER_H

/* functions */
extern void mission_manager_init(void);
extern void mission_manager_periodic(void);

/* settings */
extern float explore_forward_speed;
extern float search_forward_speed;
extern float approach_forward_speed;
extern float pass_forward_speed;
extern float recover_forward_speed;

extern float gate_confidence_threshold;
extern float gate_align_threshold;
extern float gate_approach_realign_threshold;
extern float gate_yaw_gain;
extern float search_yaw_rate_deg;

extern float pass_time_s;
extern float go_away_time_s;
extern float align_timeout_s;
extern float approach_timeout_s;
extern float avoid_timeout_s;
extern float recover_timeout_s;

#endif /* MISSION_MANAGER_H */