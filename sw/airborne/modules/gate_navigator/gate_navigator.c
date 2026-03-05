/*
 * gate_navigator.c
 *
 * Reads gate detection results from cv_detect_gate (best_gate, drone_position)
 * and flies the drone through the detected gate using GUIDED mode.
 *
 * States:
 *   SEARCH   - no gate found, rotate slowly
 *   ALIGN    - gate found, yaw to center it horizontally
 *   APPROACH - centered, fly forward using drone_position.z for distance
 *   PASS     - fly forward briefly to clear the gate
 */


#include "modules/computer_vision/snake_gate_detection.h"

#include "autopilot.h"
#include "firmwares/rotorcraft/autopilot_guided.h"

#include "modules/gate_navigator/gate_navigator.h"
#include "modules/computer_vision/detect_gate.h"   // gives us best_gate, drone_position

#include "firmwares/rotorcraft/guidance/guidance_h.h"
#include "generated/airframe.h"
#include "state.h"
#include <stdio.h>
#include <math.h>
#include "modules/datalink/datalink.h"
#include "modules/energy/electrical.h"
#include "modules/radio_control/radio_control.h"
#include "modules/ahrs/ahrs.h"

/* ── Tunable parameters (also in GCS Settings) ──────────────────────────── */
#ifndef GATE_NAV_FORWARD_SPEED
#define GATE_NAV_FORWARD_SPEED 1.0f
#endif
float nav_forward_speed  = GATE_NAV_FORWARD_SPEED;

#ifndef GATE_NAV_QUALITY_THRESH
#define GATE_NAV_QUALITY_THRESH 0.15f   // matches DETECT_GATE_MIN_GATE_QUALITY default
#endif
float nav_quality_thresh = GATE_NAV_QUALITY_THRESH;

#ifndef GATE_NAV_ALIGN_THRESH
#define GATE_NAV_ALIGN_THRESH 0.01f 
#endif
float nav_align_thresh   = GATE_NAV_ALIGN_THRESH;

#ifndef PASS_TIME_S
#define PASS_TIME_S 3.0f      // seconds to fly forward after entering gate
#endif
float pass_time_s       = PASS_TIME_S;

#ifndef GATE_NAV_SEARCH_RATE
#define GATE_NAV_SEARCH_RATE 2.5f      // deg/s rotation while searching
#endif
float nav_search_rate    = GATE_NAV_SEARCH_RATE;


/* ── Constants ───────────────────────────────────────────────────────────── */
#define IMG_WIDTH  540  // ????? dont know the resolution of the sim cam
#define IMG_HEIGHT 240

/* ── State machine ───────────────────────────────────────────────────────── */
enum gate_nav_state { SEARCH, ALIGN, APPROACH, PASS };
static enum gate_nav_state nav_state = SEARCH;

/* ── Simple timer ────────────────────────────────────────────────────────── */
static int pass_ticks = 0;

/* ── Debug mode ────────────────────────────────────────────────────────── */
#define GATE_NAV_DEBUG
#ifdef GATE_NAV_DEBUG
#define DEBUG_PRINT(fmt, ...) printf("[gate_navigator] " fmt "\n", ##__VA_ARGS__)
#else
#define DEBUG_PRINT(fmt, ...) do {} while (0)
#endif

/* ── Init ────────────────────────────────────────────────────────────────── */
void gate_navigator_init(void)
{
  nav_state  = SEARCH;
  pass_ticks = 0;
  printf("[gate_navigator] Initialized\n");
}

/* ── Periodic (10 Hz) ────────────────────────────────────────────────────── */
void gate_navigator_periodic(void)
{
  /* Only act when autopilot is in GUIDED mode */
  if (guidance_h.mode != GUIDANCE_H_MODE_GUIDED) { return; }

  /* Read detection results from cv_detect_gate globals */
  float quality  = best_gate.quality;
  int   gate_x   = best_gate.x;    // pixel x of gate center
  int   gate_sz  = best_gate.sz;   // half-size of gate in pixels
  float dist_m   = drone_position.z; // forward distance to gate in meters
  float lateral  = drone_position.x; // left(-)/right(+) offset in meters

  bool gate_found = (quality >= nav_quality_thresh);

  /* Normalized horizontal pixel error: -1 (left) .. 0 (center) .. +1 (right) */
  float h_error = (float)(gate_x - IMG_WIDTH / 2) / (IMG_WIDTH / 2.0f);

  switch (nav_state) {
    case SEARCH:
      DEBUG_PRINT("SEARCH  quality=%.2f  h_error=%.2f  dist=%.2f  gate_sz=%d", quality, h_error, dist_m, gate_sz);
      guidance_h_set_body_vel(0.f, 0.f);
      guidance_h_set_heading_rate(RadOfDeg(nav_search_rate));
      if (gate_found) { nav_state = ALIGN; }
      break;

    case ALIGN:
      DEBUG_PRINT("ALIGN   quality=%.2f  h_error=%.2f  dist=%.2f  gate_sz=%d", quality, h_error, dist_m, gate_sz);
      if (!gate_found) { 
        nav_state = SEARCH; 
        break; 
      }
      
      guidance_h_set_body_vel(0.f, 0.f);
    
      // If error is within 5% of center, stop turning and check alignment
      if (fabsf(h_error) < 0.05f) {
        guidance_h_set_heading_rate(0.f);
        nav_state = APPROACH; 
      } 
      else {
        // Slow down the turn as we get closer
        float smooth_yaw = h_error * 8.0f; 
        guidance_h_set_heading_rate(RadOfDeg(smooth_yaw));
      }
      break;

    case APPROACH:
      DEBUG_PRINT("APPROACH quality=%.2f  h_error=%.2f  dist=%.2f  gate_sz=%d", quality, h_error, dist_m, gate_sz);
      if (gate_found && (gate_sz > 75)) {
        if (fabsf(h_error) < 0.10f) {
          DEBUG_PRINT("Centered and close");
          pass_ticks = 0;
          nav_state = PASS;
          break; 
        } else {
          //re-align
          DEBUG_PRINT("Off-center, Re-aligning");
          nav_state = ALIGN;
          break;
        }
      }
      if (!gate_found) { 
        DEBUG_PRINT("Gate lost, Searching");
        nav_state = SEARCH; 
        break;
      }
      // If we drift too far off-center while flying, go back to ALIGN
      if (fabsf(h_error) > 0.125f) {
        DEBUG_PRINT("Drifted too far off-center, Re-aligning");
        nav_state = ALIGN;
        break;
      }
      // Fly forward, but use a very small correction to stay centered
      guidance_h_set_body_vel(nav_forward_speed, 0.f);
      guidance_h_set_heading_rate(RadOfDeg(h_error * 5.0f)); 
      break;

    case PASS:
      DEBUG_PRINT("PASS    quality=%.2f  h_error=%.2f  dist=%.2f  gate_sz=%d", quality, h_error, dist_m, gate_sz);
      guidance_h_set_body_vel(nav_forward_speed, 0.f);
      guidance_h_set_heading_rate(0.f);
      pass_ticks++;
      int current_pass_limit = (int)(pass_time_s * 10); // assuming this function runs at 10 Hz
      DEBUG_PRINT("PASS  tick=%d/%d", pass_ticks, current_pass_limit);

      if (pass_ticks >= current_pass_limit) {
        DEBUG_PRINT("Gate passed -> SEARCH");
        nav_state = SEARCH;
      }
      break;

    default:
      nav_state = SEARCH;
      break;
  }
}