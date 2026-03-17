/*
 * ground_seg_nav.c
 *
 * Navigation module using ground segmentation for obstacle avoidance.
 * Reads the 3x3 grid from cv_ground_seg and steers the drone toward
 * the safest direction using GUIDED mode.
 */

#include "ground_seg_nav.h"
#include "modules/ground_seg/cv_ground_seg.h"
#include "firmwares/rotorcraft/guidance/guidance_h.h"
#include "state.h"
#include <stdio.h>
#include <math.h>

#define GSN_VERBOSE TRUE
#define PRINT(string, ...) fprintf(stderr, "[ground_seg_nav->%s()] " string, __FUNCTION__, ##__VA_ARGS__)
#if GSN_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

/* State machine states */
enum gsn_nav_state_t {
  GSN_SAFE,
  GSN_OBSTACLE_FOUND,
  GSN_SEARCH_SAFE_HEADING,
  GSN_OUT_OF_BOUNDS,
  GSN_REENTER_ARENA
};

/* Tunable settings */
float gsn_max_speed     = 0.3f;   // max forward speed [m/s]
float gsn_heading_rate  = 0.4f;   // turning rate [rad/s]
float gsn_floor_frac    = 0.10f;  // min fraction of bottom row that must be ground
float gsn_obstacle_frac = 0.20f;  // min fraction of center column that must be ground

/* Internal state */
static enum gsn_nav_state_t nav_state = GSN_SEARCH_SAFE_HEADING;
static int16_t obstacle_free_confidence = 0;
static float turn_direction = 1.f;  // +1 = right (CW), -1 = left (CCW)

const int16_t max_confidence = 5;

/*
 * Decide which way to turn based on which side has more ground pixels.
 */
static void choose_turn_direction(uint32_t left_count, uint32_t right_count)
{
  if (left_count >= right_count) {
    turn_direction = -1.f;  // more ground on left -> turn left
    VERBOSE_PRINT("Turning LEFT (left=%u right=%u)\n", left_count, right_count);
  } else {
    turn_direction = 1.f;   // more ground on right -> turn right
    VERBOSE_PRINT("Turning RIGHT (left=%u right=%u)\n", left_count, right_count);
  }
}

/*
 * Init function - called once at startup
 */
void ground_seg_nav_init(void)
{
  nav_state = GSN_SEARCH_SAFE_HEADING;
  obstacle_free_confidence = 0;
  turn_direction = 1.f;
  VERBOSE_PRINT("Ground seg nav initialized\n");
}

/*
 * Periodic function - called at ~4Hz by the autopilot
 */
void ground_seg_nav_periodic(void)
{
  /* Only run in GUIDED mode */
  if (guidance_h.mode != GUIDANCE_H_MODE_GUIDED) {
    nav_state = GSN_SEARCH_SAFE_HEADING;
    obstacle_free_confidence = 0;
    return;
  }

  /* Get latest segmentation result */
  struct ground_seg_result_t seg;
  ground_seg_get_result(&seg);

  /* Use total image pixels as reference for fractions.
   * We approximate using the sum of all region counts. */
  uint32_t total_image = seg.left_count + seg.center_count + seg.right_count;
  if (total_image == 0) { total_image = 1; }  // avoid division by zero

  /* Each column/row is approximately 1/3 of total */
  uint32_t third = total_image / 3 + 1;

  /* Fraction of bottom row that is ground (cyberzoo boundary check) */
  float bottom_frac = seg.bottom_count / (float)third;

  /* Fraction of center column that is ground (obstacle check) */
  float center_frac = seg.center_count / (float)third;

  VERBOSE_PRINT("State=%d conf=%d bottom_frac=%.2f center_frac=%.2f\n",
                nav_state, obstacle_free_confidence, bottom_frac, center_frac);

  /* Update confidence: center column has enough ground = path is clear */
  if (center_frac > gsn_obstacle_frac) {
    obstacle_free_confidence++;
  } else {
    obstacle_free_confidence -= 2;
  }

  /* Clamp confidence */
  if (obstacle_free_confidence > max_confidence) { obstacle_free_confidence = max_confidence; }
  if (obstacle_free_confidence < 0)              { obstacle_free_confidence = 0; }

  /* Speed scales with confidence */
  float speed_sp = gsn_max_speed * ((float)obstacle_free_confidence / max_confidence);

  /* State machine */
  switch (nav_state) {

    case GSN_SAFE:
      if (bottom_frac < gsn_floor_frac) {
        VERBOSE_PRINT("Out of bounds detected\n");
        nav_state = GSN_OUT_OF_BOUNDS;
      } else if (obstacle_free_confidence == 0) {
        VERBOSE_PRINT("Obstacle found\n");
        nav_state = GSN_OBSTACLE_FOUND;
      } else {
        guidance_h_set_body_vel(speed_sp, 0);
      }
      break;

    case GSN_OBSTACLE_FOUND:
      /* Stop */
      guidance_h_set_body_vel(0, 0);
      /* Pick best turn direction using segmentation */
      choose_turn_direction(seg.left_count, seg.right_count);
      nav_state = GSN_SEARCH_SAFE_HEADING;
      break;

    case GSN_SEARCH_SAFE_HEADING:
      /* Keep turning until confident the path ahead is clear */
      guidance_h_set_heading_rate(turn_direction * gsn_heading_rate);
      if (obstacle_free_confidence >= 2) {
        guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi);
        nav_state = GSN_SAFE;
        VERBOSE_PRINT("Safe heading found\n");
      }
      break;

    case GSN_OUT_OF_BOUNDS:
      /* Stop and start turning back into the arena */
      guidance_h_set_body_vel(0, 0);
      guidance_h_set_heading_rate(turn_direction * gsn_heading_rate);
      nav_state = GSN_REENTER_ARENA;
      break;

    case GSN_REENTER_ARENA:
      /* Keep turning until we see enough ground in the bottom row again */
      if (bottom_frac >= gsn_floor_frac) {
        guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi);
        obstacle_free_confidence = 0;
        nav_state = GSN_SAFE;
        VERBOSE_PRINT("Re-entered arena\n");
      }
      break;

    default:
      break;
  }
}

/*
 * Dummy retreat function required by the guided flight plan.
 * The actual retreat behavior is handled by our state machine.
 */
void orange_avoider_guided_retreat(void)
{
  guidance_h_set_body_vel(-0.3f, 0);  // move backwards briefly
}