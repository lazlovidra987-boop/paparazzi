/*
 * ground_seg_nav.c
 *
 * Navigation using horizon-based ground segmentation.
 *
 * Behaviour:
 * - Move forward when the center region is sufficiently open
 * - Stop when the path is blocked
 * - Choose the side with the most visible free ground
 * - Keep turning that way until the center becomes open again
 */

#include "ground_seg_nav.h"
#include "modules/ground_seg/cv_ground_seg.h"
#include "firmwares/rotorcraft/guidance/guidance_h.h"
#include "state.h"

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

#define GSN_VERBOSE TRUE
#define PRINT(string, ...) fprintf(stderr, "[ground_seg_nav->%s()] " string, __FUNCTION__, ##__VA_ARGS__)

#if GSN_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

enum gsn_nav_state_t {
  GSN_FORWARD = 0,
  GSN_STOP_AND_DECIDE,
  GSN_TURNING
};

/* Tunable settings */
float gsn_max_speed     = 0.12f; /* forward speed [m/s] */
float gsn_heading_rate  = 0.12f; /* yaw rate while turning [rad/s] */

/*
 * Legacy names kept for settings compatibility.
 *
 * New meaning:
 * - gsn_floor_frac    = minimum center horizon mean to keep flying forward
 * - gsn_obstacle_frac = minimum center horizon mean to leave turning mode
 *
 * These values must match the actual horizon scale.
 * Your current horizon values are around 20-30, so 2-3 is too low.
 */
float gsn_floor_frac    = 10.0f;
float gsn_obstacle_frac = 14.0f;

/* Internal state */
static enum gsn_nav_state_t nav_state = GSN_STOP_AND_DECIDE;
static float turn_direction = 1.f; /* +1 = right, -1 = left */

/* Require a few stable good frames before exiting turning */
static uint8_t turn_exit_good_counter = 0U;
static const uint8_t turn_exit_good_needed = 3U;

/* -------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* -------------------------------------------------------------------------- */

static float compute_region_horizon_mean(const struct ground_seg_result_t *seg,
                                         uint16_t start, uint16_t end)
{
  if (seg->cols == 0U || end <= start || end > seg->cols) {
    return 0.f;
  }

  uint32_t sum = 0U;
  uint16_t n = 0U;

  for (uint16_t c = start; c < end; c++) {
    sum += seg->horizon[c];
    n++;
  }

  if (n == 0U) {
    return 0.f;
  }

  return sum / (float)n;
}

static float compute_center_horizon_mean(const struct ground_seg_result_t *seg)
{
  if (seg->cols == 0U) {
    return 0.f;
  }

  uint16_t start = seg->cols / 3U;
  uint16_t end   = (2U * seg->cols) / 3U;

  return compute_region_horizon_mean(seg, start, end);
}

static float compute_side_horizon_mean(const struct ground_seg_result_t *seg, bool left_side)
{
  if (seg->cols == 0U) {
    return 0.f;
  }

  if (left_side) {
    return compute_region_horizon_mean(seg, 0U, seg->cols / 3U);
  } else {
    return compute_region_horizon_mean(seg, (2U * seg->cols) / 3U, seg->cols);
  }
}

static bool is_forward_path_good(const struct ground_seg_result_t *seg)
{
  if (seg->obstacle_ahead) {
    return false;
  }

  return compute_center_horizon_mean(seg) >= gsn_floor_frac;
}

static bool is_turn_exit_condition_good(const struct ground_seg_result_t *seg)
{
  if (seg->obstacle_ahead) {
    return false;
  }

  return compute_center_horizon_mean(seg) >= gsn_obstacle_frac;
}

static void choose_turn_direction(const struct ground_seg_result_t *seg)
{
  float left_mean = compute_side_horizon_mean(seg, true);
  float right_mean = compute_side_horizon_mean(seg, false);

  /*
   * Choose the side with more visible free ground.
   * Fallback to score if means are nearly equal.
   */
  if (fabsf(left_mean - right_mean) < 1.0f) {
    if (seg->left_score >= seg->right_score) {
      turn_direction = -1.f; /* left */
      VERBOSE_PRINT("Choose LEFT by score | L=%u R=%u\n",
                    seg->left_score, seg->right_score);
    } else {
      turn_direction = 1.f;  /* right */
      VERBOSE_PRINT("Choose RIGHT by score | L=%u R=%u\n",
                    seg->left_score, seg->right_score);
    }
    return;
  }

  if (left_mean > right_mean) {
    turn_direction = -1.f; /* left */
    VERBOSE_PRINT("Choose LEFT by mean | left_mean=%.2f right_mean=%.2f\n",
                  left_mean, right_mean);
  } else {
    turn_direction = 1.f;  /* right */
    VERBOSE_PRINT("Choose RIGHT by mean | left_mean=%.2f right_mean=%.2f\n",
                  left_mean, right_mean);
  }
}

static void hold_current_heading(void)
{
  guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi);
}

static void stop_motion(void)
{
  guidance_h_set_body_vel(0.f, 0.f);
}

static void command_forward(void)
{
  hold_current_heading();
  guidance_h_set_body_vel(gsn_max_speed, 0.f);
}

static void command_turn(void)
{
  stop_motion();
  guidance_h_set_heading_rate(turn_direction * gsn_heading_rate);
}

/* -------------------------------------------------------------------------- */
/* Paparazzi hooks                                                            */
/* -------------------------------------------------------------------------- */

void ground_seg_nav_init(void)
{
  nav_state = GSN_STOP_AND_DECIDE;
  turn_direction = 1.f;
  turn_exit_good_counter = 0U;
  VERBOSE_PRINT("Ground seg nav initialized\n");
}

void ground_seg_nav_periodic(void)
{
  if (guidance_h.mode != GUIDANCE_H_MODE_GUIDED) {
    nav_state = GSN_STOP_AND_DECIDE;
    turn_exit_good_counter = 0U;
    stop_motion();
    return;
  }

  struct ground_seg_result_t seg;
  ground_seg_get_result(&seg);

  if (seg.cols == 0U || seg.rows == 0U) {
    VERBOSE_PRINT("No valid segmentation data\n");
    turn_exit_good_counter = 0U;
    stop_motion();
    return;
  }

  float center_mean = compute_center_horizon_mean(&seg);
  float left_mean   = compute_side_horizon_mean(&seg, true);
  float right_mean  = compute_side_horizon_mean(&seg, false);

  bool forward_ok   = is_forward_path_good(&seg);
  bool turn_exit_ok = is_turn_exit_condition_good(&seg);

  VERBOSE_PRINT("State=%d obstacle=%d center_mean=%.2f left_mean=%.2f right_mean=%.2f | L=%u C=%u R=%u | turn_dir=%.1f\n",
                nav_state,
                seg.obstacle_ahead,
                center_mean,
                left_mean,
                right_mean,
                seg.left_score,
                seg.center_score,
                seg.right_score,
                turn_direction);

  switch (nav_state) {

    case GSN_FORWARD:
      turn_exit_good_counter = 0U;

      if (forward_ok) {
        command_forward();
      } else {
        stop_motion();
        nav_state = GSN_STOP_AND_DECIDE;
        VERBOSE_PRINT("Forward blocked -> stop and decide\n");
      }
      break;

    case GSN_STOP_AND_DECIDE:
      stop_motion();
      turn_exit_good_counter = 0U;

      if (forward_ok) {
        nav_state = GSN_FORWARD;
        VERBOSE_PRINT("Forward path clear -> move forward\n");
      } else {
        choose_turn_direction(&seg);
        nav_state = GSN_TURNING;
        VERBOSE_PRINT("Start turning\n");
      }
      break;

    case GSN_TURNING:
      command_turn();

      if (turn_exit_ok) {
        if (turn_exit_good_counter < 255U) {
          turn_exit_good_counter++;
        }
      } else {
        turn_exit_good_counter = 0U;
      }

      if (turn_exit_good_counter >= turn_exit_good_needed) {
        hold_current_heading();
        nav_state = GSN_FORWARD;
        turn_exit_good_counter = 0U;
        VERBOSE_PRINT("Center open again -> forward\n");
      }
      break;

    default:
      stop_motion();
      nav_state = GSN_STOP_AND_DECIDE;
      turn_exit_good_counter = 0U;
      break;
  }
}

/*
 * Kept for guided flight-plan compatibility.
 * Real avoidance is handled by the state machine above.
 */
void orange_avoider_guided_retreat(void)
{
  guidance_h_set_body_vel(-0.2f, 0.f);
}