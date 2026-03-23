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
//#include "modules/your_cnn_module/your_cnn_module.h"  /* TODO: find real path */

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
/* --- Gate --- */
float gsn_gate_conf_low             = 0.40f;  /* Below this, ignore gate detection */
float gsn_gate_conf_high            = 0.70f;  /* Minimum confidence for a valid gate candidate */
float gsn_gate_approach_speed       = 0.18f;  /* Speed when confidently approaching a gate */
float gsn_gate_max_heading_error    = 0.6f;   /* Maximum allowed heading error to approach the gate [rad] */
float gsn_gate_heading_filter_alpha = 0.7f;   /* Low-pass filter alpha for gate heading (0-1, higher is smoother) */

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

/*
 * Gate detection internal state.
 *
 * Heading values:
 * - gate_heading           = raw heading error from CNN [rad], updated each cycle when valid
 * - gate_heading_filtered  = low-pass smoothed heading, used for all control commands
 * - last_good_gate_heading = last filtered heading with high confidence,
 *                            used as fallback during GSN_GATE_LOST_RECOVER
 *
 * Confidence:
 * - gate_confidence        = raw confidence score from CNN [0.0 - 1.0], updated each cycle
 *
 * Confirmation counters:
 * - gate_seen_counter      = consecutive cycles with a good detection,
 *                            must reach gate_seen_needed before entering GSN_GATE_APPROACH
 * - gate_lost_counter      = consecutive cycles without a good detection,
 *                            triggers return to GSN_FORWARD when it reaches gate_lost_max
 *
 * Thresholds:
 * - gate_seen_needed       = number of stable frames required to confirm a gate (enter approach)
 * - gate_lost_max          = number of missing frames tolerated before abandoning gate pursuit
 *
 * Higher gate_seen_needed = less likely to chase false positives, slower to react
 * Higher gate_lost_max    = more tolerant of brief CNN dropouts, slower to give up
 */
static float   gate_heading           = 0.f;
static float   gate_heading_filtered  = 0.f;
static float   gate_confidence        = 0.f;
static float   last_good_gate_heading = 0.f;

static uint8_t gate_seen_counter = 0U;
static uint8_t gate_lost_counter = 0U;

static const uint8_t gate_seen_needed = 3U;
static const uint8_t gate_lost_max    = 5U;

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


/* ----- Gate helpers ----- */

/* TODO: confirm variable names */
static void get_gate_result(float *heading, float *confidence, bool *valid)
{
  *heading    = gate_cnn_result.heading;
  *confidence = gate_cnn_result.conf;
  *valid      = (gate_cnn_result.conf > 0.f); /* TODO: confirm validity condition with your CNN module */
}


static bool is_gate_conf_low(float conf)  { return conf <  gsn_gate_conf_low;  }  /* Below this, ignore gate detection */
static bool is_gate_conf_high(float conf) { return conf >= gsn_gate_conf_high; }  /* Minimum confidence for a valid gate candidate */

/* Check if CNN heading is valid */
static bool is_gate_heading_valid(float heading)  /* Check if the heading error is a reasonable value (not NaN/inf and within max error) */
{
  if (isnan(heading) || isinf(heading)) { return false; }
  return fabsf(heading) <= gsn_gate_max_heading_error * 5.f;
}

/* Low-pass filter for gate heading */
static float filter_gate_heading(float prev, float raw)
{
  return gsn_gate_heading_filter_alpha * prev
       + (1.f - gsn_gate_heading_filter_alpha) * raw;
}

/* Check confidence and heading validity */
static bool is_gate_candidate_good(bool valid, float heading, float conf)
{
  return valid
      && is_gate_conf_high(conf)
      && is_gate_heading_valid(heading);
}

/* Check if we can safely approach the gate based on current segmentation */
static bool is_gate_approach_safe(const struct ground_seg_result_t *seg)
{
  return is_forward_path_good(seg);
}

/*
 * -- TODO: confirm control logic with your CNN output --

 * gate_heading_offset is a body-frame heading error [rad].
 * Positive = gate is right of drone nose, negative = left.
 * If CNN outputs absolute NED heading, remove the current_heading addition.
 */
static void command_gate_approach(float gate_heading_offset)
{
  float desired = stateGetNedToBodyEulers_f()->psi + gate_heading_offset;
  guidance_h_set_heading(desired);
  guidance_h_set_body_vel(gsn_gate_approach_speed, 0.f);
}

/*
 * Kept for guided flight-plan compatibility.
 * Real avoidance is handled by the state machine above.
 */
void orange_avoider_guided_retreat(void)
{
  guidance_h_set_body_vel(-0.2f, 0.f);
}