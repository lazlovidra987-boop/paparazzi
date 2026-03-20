/*
 * mission_manager.c
 * -----------------
 * High-level mission executive for:
 *  - exploring for distance (position-aware heading bias)
 *  - avoiding obstacles (orange pixel count via ABI, same pattern as orange_avoider.c)
 *  - recovering from boundary violations (InsideObstacleZone / InsideCyberZoo)
 *  - finding and flying through a gate using CNN
 *  - repeating gate passes (gate moves after each pass)
 */

#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

#include "state.h"
#include "generated/airframe.h"
#include "autopilot.h"

#include "firmwares/rotorcraft/guidance/guidance_h.h"
#include "firmwares/rotorcraft/guidance/guidance_v.h"

#include "modules/gate_cnn/gate_cnn.h"
#include "modules/core/abi.h"
#include "generated/flight_plan.h"


/* ============================================================
 * Tunable parameters
 * ============================================================ */

#ifndef MISSION_PERIODIC_HZ
#define MISSION_PERIODIC_HZ 10
#endif

/* Flight speeds */
#ifndef EXPLORE_FORWARD_SPEED
#define EXPLORE_FORWARD_SPEED 0.8f
#endif
float explore_forward_speed = EXPLORE_FORWARD_SPEED;

#ifndef SEARCH_FORWARD_SPEED
#define SEARCH_FORWARD_SPEED 0.5f
#endif
float search_forward_speed = SEARCH_FORWARD_SPEED;

#ifndef APPROACH_FORWARD_SPEED
#define APPROACH_FORWARD_SPEED 0.7f
#endif
float approach_forward_speed = APPROACH_FORWARD_SPEED;

#ifndef PASS_FORWARD_SPEED
#define PASS_FORWARD_SPEED 0.9f
#endif
float pass_forward_speed = PASS_FORWARD_SPEED;

#ifndef RECOVER_FORWARD_SPEED
#define RECOVER_FORWARD_SPEED 0.5f
#endif
float recover_forward_speed = RECOVER_FORWARD_SPEED;

/* Gate detection */
#ifndef GATE_CONFIDENCE_THRESHOLD
#define GATE_CONFIDENCE_THRESHOLD 0.6f
#endif
float gate_confidence_threshold = GATE_CONFIDENCE_THRESHOLD;

#ifndef GATE_ALIGN_THRESHOLD
#define GATE_ALIGN_THRESHOLD 0.08f
#endif
float gate_align_threshold = GATE_ALIGN_THRESHOLD;

#ifndef GATE_APPROACH_REALIGN_THRESHOLD
#define GATE_APPROACH_REALIGN_THRESHOLD 0.20f
#endif
float gate_approach_realign_threshold = GATE_APPROACH_REALIGN_THRESHOLD;

#ifndef GATE_YAW_GAIN
#define GATE_YAW_GAIN 15.0f
#endif
float gate_yaw_gain = GATE_YAW_GAIN;

#ifndef SEARCH_YAW_RATE_DEG
#define SEARCH_YAW_RATE_DEG 6.0f
#endif
float search_yaw_rate_deg = SEARCH_YAW_RATE_DEG;

/* Timing */
#ifndef PASS_TIME_S
#define PASS_TIME_S 2.0f
#endif
float pass_time_s = PASS_TIME_S;

#ifndef GO_AWAY_TIME_S
#define GO_AWAY_TIME_S 1.5f
#endif
float go_away_time_s = GO_AWAY_TIME_S;

#ifndef ALIGN_TIMEOUT_S
#define ALIGN_TIMEOUT_S 3.0f
#endif
float align_timeout_s = ALIGN_TIMEOUT_S;

#ifndef APPROACH_TIMEOUT_S
#define APPROACH_TIMEOUT_S 8.0f
#endif
float approach_timeout_s = APPROACH_TIMEOUT_S;

#ifndef AVOID_TIMEOUT_S
#define AVOID_TIMEOUT_S 3.0f
#endif
float avoid_timeout_s = AVOID_TIMEOUT_S;

#ifndef RECOVER_TIMEOUT_S
#define RECOVER_TIMEOUT_S 4.0f
#endif
float recover_timeout_s = RECOVER_TIMEOUT_S;

/* Obstacle avoidance — pixel fraction threshold (same as orange_avoider) */
#ifndef MISSION_OA_COLOR_FRAC
#define MISSION_OA_COLOR_FRAC 0.18f
#endif
static float oa_color_count_frac = MISSION_OA_COLOR_FRAC;

/*
 * How many consecutive 10 Hz ticks must be clear before we declare the
 * obstacle gone.  5 ticks = 0.5 s, matches orange_avoider's max_trajectory_confidence.
 */
#ifndef OBSTACLE_CLEAR_HYSTERESIS
#define OBSTACLE_CLEAR_HYSTERESIS 5
#endif

/*
 * How many consecutive ticks must show the gate before we call it "stable".
 * Similarly, how many consecutive misses before we call it "lost".
 */
#ifndef GATE_STABLE_COUNT
#define GATE_STABLE_COUNT 3
#endif
#ifndef GATE_LOST_COUNT
#define GATE_LOST_COUNT 5
#endif

/*
 * How many consecutive ticks must be aligned before committing to PASS.
 */
#ifndef GATE_COMMIT_COUNT
#define GATE_COMMIT_COUNT 3
#endif

/*
 * Exploration heading bias: how many degrees per second we nudge
 * preferred_heading toward the far side of the arena.
 */
#ifndef EXPLORE_HEADING_BIAS_DEG_S
#define EXPLORE_HEADING_BIAS_DEG_S 2.0f
#endif


/* ============================================================
 * ABI obstacle detection  (mirrors orange_avoider.c exactly)
 * ============================================================ */

#ifndef MISSION_VISUAL_DETECTION_ID
#define MISSION_VISUAL_DETECTION_ID ABI_BROADCAST
#endif

static abi_event color_detection_ev;
static int32_t   color_count = 0;   /* updated asynchronously by ABI callback */

static void color_detection_cb(uint8_t  __attribute__((unused)) sender_id,
                                int16_t  __attribute__((unused)) pixel_x,
                                int16_t  __attribute__((unused)) pixel_y,
                                int16_t  __attribute__((unused)) pixel_width,
                                int16_t  __attribute__((unused)) pixel_height,
                                int32_t  quality,
                                int16_t  __attribute__((unused)) extra)
{
  color_count = quality;
}


/* ============================================================
 * Debug
 * ============================================================ */

#define MISSION_DEBUG 1

#if MISSION_DEBUG
#define DEBUG_PRINT(fmt, ...) \
  printf("[mission_manager] " fmt "\n", ##__VA_ARGS__)
#else
#define DEBUG_PRINT(fmt, ...) do {} while (0)
#endif


/* ============================================================
 * State machine
 * ============================================================ */

enum mission_state_t {
  STARTUP,
  EXPLORE,
  SEARCH_GATE,
  ALIGN_GATE,
  APPROACH_GATE,
  PASS_GATE,
  GO_AWAY,
  AVOID_OBSTACLE,
  RECOVER_BOUNDS,
  RECOVERY
};


/* ============================================================
 * Status / context
 * ============================================================ */

struct mission_status_t {
  bool in_guided_mode;
  bool in_flight;

  bool near_bounds;
  bool out_of_bounds;
  bool obstacle_imminent;

  bool gate_visible;
  bool gate_stable;
  bool gate_aligned;

  float gate_heading_error;
  float gate_confidence;

  float pos_x;
  float pos_y;
  float heading;
};

struct mission_context_t {
  enum mission_state_t state;
  enum mission_state_t prev_state;

  int state_ticks;

  int gate_seen_count;
  int gate_lost_count;
  int align_good_count;

  int obstacle_seen_count;
  int obstacle_clear_count;
  int16_t obstacle_free_confidence;   /* same semantics as orange_avoider */

  int bounds_safe_count;

  float preferred_heading;
  float recovery_heading;
  float go_away_heading;

  /*
   * Last known gate heading error — stored even when gate is not visible.
   * Used in SEARCH_GATE to bias rotation toward where we last saw the gate,
   * instead of always spinning in the same direction.
   */
  float gate_last_heading;
  bool  gate_ever_seen;

  bool state_entry;
};


/* ============================================================
 * Global module state
 * ============================================================ */

static struct mission_status_t status;
static struct mission_context_t ctx;


/* ============================================================
 * Forward declarations
 * ============================================================ */

void mission_manager_init(void);
void mission_manager_periodic(void);

static void update_status(void);
static void update_gate_status(void);
static void update_bounds_status(void);
static void update_obstacle_status(void);
static void update_position_status(void);

static void step_state_machine(void);
static void change_state(enum mission_state_t new_state);
static const char *state_name(enum mission_state_t s);

static bool handle_global_safety_transitions(void);

static void run_startup(void);
static void run_explore(void);
static void run_search_gate(void);
static void run_align_gate(void);
static void run_approach_gate(void);
static void run_pass_gate(void);
static void run_go_away(void);
static void run_avoid_obstacle(void);
static void run_recover_bounds(void);
static void run_recovery(void);

static void command_stop(void);
static void command_forward_heading_rate(float forward_speed, float heading_rate_rad_s);
static void command_forward_heading_hold(float forward_speed, float heading_rad);
static void command_hover_heading_hold(float heading_rad) __attribute__((unused));

static bool  state_timed_out(float timeout_s);
static float compute_inward_heading(void);
static float wrap_angle(float a);
static float far_side_heading(void);


/* ============================================================
 * Init
 * ============================================================ */

void mission_manager_init(void)
{
  /* Register ABI listener for the color filter output — same pattern as
   * orange_avoider.c.  The color filter module publishes VISUAL_DETECTION
   * messages; we bind here so mission_manager is self-contained. */
  AbiBindMsgVISUAL_DETECTION(MISSION_VISUAL_DETECTION_ID,
                              &color_detection_ev,
                              color_detection_cb);

  ctx.state       = STARTUP;
  ctx.prev_state  = STARTUP;
  ctx.state_ticks = 0;
  ctx.state_entry = true;

  ctx.gate_seen_count  = 0;
  ctx.gate_lost_count  = 0;
  ctx.align_good_count = 0;

  ctx.obstacle_seen_count      = 0;
  ctx.obstacle_clear_count     = 0;
  ctx.obstacle_free_confidence = 0;

  ctx.bounds_safe_count = 0;

  ctx.preferred_heading = 0.0f;
  ctx.recovery_heading  = 0.0f;
  ctx.go_away_heading   = 0.0f;

  ctx.gate_last_heading = 0.0f;
  ctx.gate_ever_seen    = false;

  DEBUG_PRINT("Initialised  oa_frac=%.2f  gate_conf_thresh=%.2f",
              oa_color_count_frac, gate_confidence_threshold);
}


/* ============================================================
 * Periodic (called at MISSION_PERIODIC_HZ)
 * ============================================================ */

void mission_manager_periodic(void)
{
  update_status();

  if (!status.in_guided_mode) {
    return;
  }

  /* Hold altitude — vertical guidance is not mission-managed */
  guidance_v_set_vz(0.0f);

  step_state_machine();
}


/* ============================================================
 * Status updates
 * ============================================================ */

static void update_status(void)
{
  update_position_status();
  update_gate_status();
  update_bounds_status();
  update_obstacle_status();

  status.in_guided_mode = (guidance_h.mode == GUIDANCE_H_MODE_GUIDED);
  status.in_flight      = autopilot_in_flight();
}

static void update_position_status(void)
{
  status.pos_x   = stateGetPositionEnu_f()->x;
  status.pos_y   = stateGetPositionEnu_f()->y;
  status.heading = stateGetNedToBodyEulers_f()->psi;
}

static void update_gate_status(void)
{
  status.gate_heading_error = gate_cnn_result.heading;
  status.gate_confidence    = gate_cnn_result.gate_measure;
  status.gate_visible       = (status.gate_confidence >= gate_confidence_threshold);

  if (status.gate_visible) {
    /* Store last known heading for directed re-acquisition in SEARCH_GATE */
    ctx.gate_last_heading = status.gate_heading_error;
    ctx.gate_ever_seen    = true;

    ctx.gate_seen_count++;
    ctx.gate_lost_count = 0;
  } else {
    ctx.gate_lost_count++;
    ctx.gate_seen_count = 0;
  }

  /* Symmetric hysteresis: require GATE_STABLE_COUNT consecutive detections
   * to declare stable, and GATE_LOST_COUNT consecutive misses to declare lost.
   * This prevents a single bad CNN frame from abandoning an alignment. */
  status.gate_stable = (ctx.gate_seen_count  >= GATE_STABLE_COUNT);

  /* gate_visible used for "lost" transitions is overridden here with hysteresis */
  if (ctx.gate_lost_count >= GATE_LOST_COUNT) {
    status.gate_visible = false;
  } else if (ctx.gate_seen_count >= 1) {
    status.gate_visible = true;
  }

  status.gate_aligned = (fabsf(status.gate_heading_error) < gate_align_threshold);

  if (status.gate_aligned && status.gate_visible) {
    ctx.align_good_count++;
  } else {
    ctx.align_good_count = 0;
  }
}

static void update_bounds_status(void)
{
  /*
   * Use the flight-plan sector macros, exactly as orange_avoider.c uses
   * InsideObstacleZone().  InsideCyberZoo() is the hard outer boundary;
   * InsideObstacleZone() is the slightly smaller inner safe zone.
   *
   * out_of_bounds  = outside the hard CyberZoo boundary
   * near_bounds    = inside CyberZoo but outside the soft ObstacleZone
   *
   * This gives us a two-stage warning: near_bounds triggers a gentle
   * inward heading correction; out_of_bounds triggers a forced recovery.
   */
  float x = status.pos_x;
  float y = status.pos_y;

  status.out_of_bounds = !InsideCyberZoo(x, y);
  status.near_bounds   = !status.out_of_bounds && !InsideObstacleZone(x, y);

  if (!status.out_of_bounds && !status.near_bounds) {
    ctx.bounds_safe_count++;
  } else {
    ctx.bounds_safe_count = 0;
  }
}

static void update_obstacle_status(void)
{
  /*
   * Mirrors orange_avoider.c exactly:
   *  - compare color_count against a fraction of total pixels
   *  - obstacle_free_confidence increments on clear frames, decrements by 2
   *    on detected frames (asymmetric: faster to trigger than to clear)
   *  - bounded to [0, OBSTACLE_CLEAR_HYSTERESIS]
   *  - obstacle_imminent when confidence hits 0
   *
   * front_camera.output_size.w/h is the same camera reference used in
   * orange_avoider.c.
   */
  int32_t threshold = (int32_t)(oa_color_count_frac
                                * front_camera.output_size.w
                                * front_camera.output_size.h);

  if (color_count < threshold) {
    ctx.obstacle_free_confidence++;
  } else {
    ctx.obstacle_free_confidence -= 2;
  }

  /* Bound to [0, OBSTACLE_CLEAR_HYSTERESIS] */
  if (ctx.obstacle_free_confidence < 0) {
    ctx.obstacle_free_confidence = 0;
  }
  if (ctx.obstacle_free_confidence > OBSTACLE_CLEAR_HYSTERESIS) {
    ctx.obstacle_free_confidence = OBSTACLE_CLEAR_HYSTERESIS;
  }

  status.obstacle_imminent = (ctx.obstacle_free_confidence == 0);

  /* Legacy counters kept for transition logic */
  if (status.obstacle_imminent) {
    ctx.obstacle_seen_count++;
    ctx.obstacle_clear_count = 0;
  } else {
    ctx.obstacle_clear_count++;
    ctx.obstacle_seen_count = 0;
  }
}


/* ============================================================
 * State machine core
 * ============================================================ */

static void step_state_machine(void)
{
  ctx.state_ticks++;

  /*
   * Safety transitions are checked first.  If a transition fires we return
   * immediately — the current state handler does NOT run this tick.
   * (Previously the code fell through into the switch; that was a bug.)
   */
  if (handle_global_safety_transitions()) {
    /* change_state() already set state_entry = true for the new state.
     * Do NOT clear it here — just return so the new state runs next tick. */
    return;
  }

  switch (ctx.state) {
    case STARTUP:         run_startup();         break;
    case EXPLORE:         run_explore();         break;
    case SEARCH_GATE:     run_search_gate();     break;
    case ALIGN_GATE:      run_align_gate();      break;
    case APPROACH_GATE:   run_approach_gate();   break;
    case PASS_GATE:       run_pass_gate();       break;
    case GO_AWAY:         run_go_away();         break;
    case AVOID_OBSTACLE:  run_avoid_obstacle();  break;
    case RECOVER_BOUNDS:  run_recover_bounds();  break;
    case RECOVERY:        run_recovery();        break;
    default:              change_state(RECOVERY); break;
  }

  ctx.state_entry = false;
}

static bool handle_global_safety_transitions(void)
{
  /*
   * PASS_GATE is intentionally excluded from both safety overrides:
   * - The drone is already committed and mid-pass; aborting is more
   *   dangerous than completing the 2-second straight-line dash.
   * - An obstacle detection *inside* the gate frame is almost certainly
   *   a false positive (orange gate structure in frame).
   */

  /* Highest priority: boundary violation */
  if (ctx.state != PASS_GATE      &&
      ctx.state != RECOVER_BOUNDS &&
      ctx.state != RECOVERY) {
    if (status.out_of_bounds || status.near_bounds) {
      change_state(RECOVER_BOUNDS);
      return true;
    }
  }

  /* Second priority: imminent obstacle */
  if (ctx.state != PASS_GATE     &&
      ctx.state != AVOID_OBSTACLE &&
      ctx.state != RECOVERY       &&
      ctx.state != RECOVER_BOUNDS) {
    if (status.obstacle_imminent) {
      change_state(AVOID_OBSTACLE);
      return true;
    }
  }

  return false;
}

static void change_state(enum mission_state_t new_state)
{
  if (new_state == ctx.state) {
    return;
  }

  ctx.prev_state  = ctx.state;
  ctx.state       = new_state;
  ctx.state_ticks = 0;
  ctx.state_entry = true;

  DEBUG_PRINT("State: %s -> %s  pos=(%.2f, %.2f)  hdg=%.1f deg",
              state_name(ctx.prev_state), state_name(ctx.state),
              status.pos_x, status.pos_y,
              (double)status.heading * 180.0 / M_PI);
}

static const char *state_name(enum mission_state_t s)
{
  switch (s) {
    case STARTUP:         return "STARTUP";
    case EXPLORE:         return "EXPLORE";
    case SEARCH_GATE:     return "SEARCH_GATE";
    case ALIGN_GATE:      return "ALIGN_GATE";
    case APPROACH_GATE:   return "APPROACH_GATE";
    case PASS_GATE:       return "PASS_GATE";
    case GO_AWAY:         return "GO_AWAY";
    case AVOID_OBSTACLE:  return "AVOID_OBSTACLE";
    case RECOVER_BOUNDS:  return "RECOVER_BOUNDS";
    case RECOVERY:        return "RECOVERY";
    default:              return "UNKNOWN";
  }
}


/* ============================================================
 * State handlers
 * ============================================================ */

static void run_startup(void)
{
  if (ctx.state_entry) {
    ctx.preferred_heading = status.heading;
    DEBUG_PRINT("Entering STARTUP  preferred_heading=%.1f deg",
                (double)ctx.preferred_heading * 180.0 / M_PI);
  }

  command_stop();
  change_state(EXPLORE);
}

/* ----------------------------------------------------------
 * EXPLORE
 * Fly forward on preferred_heading, slowly biasing that heading
 * toward the far side of the arena using OptiTrack position.
 * Switch to SEARCH_GATE after a timeout if no gate is found,
 * or immediately if the gate becomes stably visible.
 * ---------------------------------------------------------- */
static void run_explore(void)
{
  if (ctx.state_entry) {
    DEBUG_PRINT("Entering EXPLORE  pos=(%.2f, %.2f)",
                status.pos_x, status.pos_y);
  }

  /* Gate found while exploring — go directly to alignment */
  if (status.gate_stable) {
    change_state(ALIGN_GATE);
    return;
  }

  /*
   * Position-aware heading bias.
   * far_side_heading() returns the heading toward the half of the arena
   * that is furthest from the current position.  We nudge preferred_heading
   * toward it at EXPLORE_HEADING_BIAS_DEG_S degrees per second, so the
   * drone gradually sweeps the arena rather than driving into a wall.
   */
  float target  = far_side_heading();
  float diff    = wrap_angle(target - ctx.preferred_heading);
  float max_step = RadOfDeg(EXPLORE_HEADING_BIAS_DEG_S / MISSION_PERIODIC_HZ);

  if (fabsf(diff) > max_step) {
    ctx.preferred_heading = wrap_angle(ctx.preferred_heading +
                                       (diff > 0 ? max_step : -max_step));
  } else {
    ctx.preferred_heading = target;
  }

  command_forward_heading_hold(explore_forward_speed, ctx.preferred_heading);

  /* After 4 s without finding the gate, enter a dedicated search sweep */
  if (state_timed_out(4.0f)) {
    change_state(SEARCH_GATE);
    return;
  }
}

/* ----------------------------------------------------------
 * SEARCH_GATE
 * Rotate slowly.  If we have ever seen the gate, bias the
 * rotation direction toward gate_last_heading so we re-acquire
 * it quickly rather than always spinning the same way.
 * ---------------------------------------------------------- */
static void run_search_gate(void)
{
  if (ctx.state_entry) {
    DEBUG_PRINT("Entering SEARCH_GATE  ever_seen=%d  last_hdg_err=%.3f",
                ctx.gate_ever_seen, ctx.gate_last_heading);
  }

  if (status.gate_stable) {
    change_state(ALIGN_GATE);
    return;
  }

  /*
   * Directed search: if we have a last-known heading error, rotate toward
   * where the gate was last seen.  Otherwise rotate at the default rate.
   */
  float rate;
  if (ctx.gate_ever_seen && fabsf(ctx.gate_last_heading) > 0.05f) {
    /* Rotate toward last known gate side, at double the search rate to
     * re-acquire faster */
    rate = (ctx.gate_last_heading > 0.0f)
           ?  RadOfDeg(search_yaw_rate_deg * 2.0f)
           : -RadOfDeg(search_yaw_rate_deg * 2.0f);
  } else {
    rate = RadOfDeg(search_yaw_rate_deg);
  }

  command_forward_heading_rate(search_forward_speed, rate);

  /* After 4 s of searching, return to explore so we also change position */
  if (state_timed_out(4.0f)) {
    change_state(EXPLORE);
    return;
  }
}

/* ----------------------------------------------------------
 * ALIGN_GATE
 * Hover and yaw until heading error is below threshold for
 * GATE_COMMIT_COUNT consecutive ticks, then approach.
 * ---------------------------------------------------------- */
static void run_align_gate(void)
{
  if (ctx.state_entry) {
    DEBUG_PRINT("Entering ALIGN_GATE");
  }

  /* Hysteresis-protected gate-lost check */
  if (!status.gate_visible) {
    change_state(SEARCH_GATE);
    return;
  }

  /* Proportional yaw toward gate centre, no forward motion */
  command_forward_heading_rate(
    0.0f,
    RadOfDeg(status.gate_heading_error * gate_yaw_gain)
  );

  /* Commit condition: aligned for GATE_COMMIT_COUNT consecutive ticks */
  if (ctx.align_good_count >= GATE_COMMIT_COUNT) {
    change_state(APPROACH_GATE);
    return;
  }

  if (state_timed_out(align_timeout_s)) {
    DEBUG_PRINT("ALIGN_GATE timed out");
    change_state(SEARCH_GATE);
    return;
  }
}

/* ----------------------------------------------------------
 * APPROACH_GATE
 * Fly forward while tracking the gate heading.  Re-align if
 * heading error grows too large.  Commit to PASS_GATE once
 * the gate has been stably aligned for GATE_COMMIT_COUNT ticks
 * (replaces the old unreachable 1.5 s hard trigger).
 * ---------------------------------------------------------- */
static void run_approach_gate(void)
{
  if (ctx.state_entry) {
    DEBUG_PRINT("Entering APPROACH_GATE");
    /* Reset so we don't inherit a high count from ALIGN_GATE and
     * instantly commit to PASS_GATE on the first tick. */
    ctx.align_good_count = 0;
  }

  /* Hysteresis-protected gate-lost check */
  if (!status.gate_visible) {
    command_stop();
    change_state(SEARCH_GATE);
    return;
  }

  /* Large drift — stop and re-centre before continuing */
  if (fabsf(status.gate_heading_error) > gate_approach_realign_threshold) {
    command_stop();
    change_state(ALIGN_GATE);
    return;
  }

  /* Commit condition: stably aligned → commit to pass */
  if (status.gate_aligned && ctx.align_good_count >= GATE_COMMIT_COUNT) {
    change_state(PASS_GATE);
    return;
  }

  /* Hard timeout — gate approach is taking too long, go back to search */
  if (state_timed_out(approach_timeout_s)) {
    DEBUG_PRINT("APPROACH_GATE timed out");
    command_stop();
    change_state(SEARCH_GATE);
    return;
  }

  /* Fly forward with gentle heading correction */
  command_forward_heading_rate(
    approach_forward_speed,
    RadOfDeg(status.gate_heading_error * gate_yaw_gain * 0.4f)
  );
}

/* ----------------------------------------------------------
 * PASS_GATE
 * Commit: fly straight through at full speed.  No corrections,
 * no safety overrides (see handle_global_safety_transitions).
 * ---------------------------------------------------------- */
static void run_pass_gate(void)
{
  if (ctx.state_entry) {
    DEBUG_PRINT("Entering PASS_GATE  commit_heading=%.1f deg",
                (double)status.heading * 180.0 / M_PI);
    /* Lock in the heading at the moment we commit */
    ctx.go_away_heading = status.heading;
  }

  command_forward_heading_hold(pass_forward_speed, ctx.go_away_heading);

  if (state_timed_out(pass_time_s)) {
    DEBUG_PRINT("Gate passed!");
    change_state(GO_AWAY);
    return;
  }
}

/* ----------------------------------------------------------
 * GO_AWAY
 * Continue forward briefly to clear the gate structure, then
 * return to EXPLORE.  The gate will have been moved by then,
 * so EXPLORE will eventually lead back to a new SEARCH_GATE.
 * Reset gate_ever_seen so SEARCH_GATE doesn't chase the old
 * last-seen direction.
 * ---------------------------------------------------------- */
static void run_go_away(void)
{
  if (ctx.state_entry) {
    DEBUG_PRINT("Entering GO_AWAY");
    /* Forget old gate position — it will be moved */
    ctx.gate_ever_seen = false;
    ctx.gate_last_heading = 0.0f;
  }

  command_forward_heading_hold(approach_forward_speed, ctx.go_away_heading);

  if (state_timed_out(go_away_time_s)) {
    ctx.preferred_heading = status.heading;
    change_state(EXPLORE);
    return;
  }
}

/* ----------------------------------------------------------
 * AVOID_OBSTACLE
 * Rotate away from the obstacle.  Direction is fixed per entry
 * (currently always CCW — TODO: use pixel centroid when available).
 * Once the confidence recovers, resume EXPLORE.
 * ---------------------------------------------------------- */
static void run_avoid_obstacle(void)
{
  if (ctx.state_entry) {
    DEBUG_PRINT("Entering AVOID_OBSTACLE  conf=%d", ctx.obstacle_free_confidence);
  }

  /* Rotate in place — 20 deg/s CCW.
   * V1: direction is fixed.  In V2 replace with pixel centroid logic:
   *   if (obstacle_centroid_x > frame_width/2) rotate CCW else CW
   */
  command_forward_heading_rate(0.0f, RadOfDeg(20.0f));

  /* Clear: confidence has recovered above the trigger level */
  if (!status.obstacle_imminent &&
      ctx.obstacle_clear_count >= OBSTACLE_CLEAR_HYSTERESIS) {
    ctx.preferred_heading = status.heading;
    change_state(EXPLORE);
    return;
  }

  if (state_timed_out(avoid_timeout_s)) {
    DEBUG_PRINT("AVOID_OBSTACLE timed out -> RECOVERY");
    change_state(RECOVERY);
    return;
  }
}

/* ----------------------------------------------------------
 * RECOVER_BOUNDS
 * Fly toward the arena centre at moderate speed.  Once safely
 * inside ObstacleZone for bounds_safe_count ticks, resume EXPLORE.
 * ---------------------------------------------------------- */
static void run_recover_bounds(void)
{
  if (ctx.state_entry) {
    DEBUG_PRINT("Entering RECOVER_BOUNDS  pos=(%.2f, %.2f)",
                status.pos_x, status.pos_y);
    ctx.recovery_heading = compute_inward_heading();
  }

  command_forward_heading_hold(recover_forward_speed, ctx.recovery_heading);

  if (!status.out_of_bounds && !status.near_bounds &&
      ctx.bounds_safe_count >= 3) {
    ctx.preferred_heading = status.heading;
    change_state(EXPLORE);
    return;
  }

  if (state_timed_out(recover_timeout_s)) {
    DEBUG_PRINT("RECOVER_BOUNDS timed out -> RECOVERY");
    change_state(RECOVERY);
    return;
  }
}

/* ----------------------------------------------------------
 * RECOVERY
 * Last-resort state: slow flight toward arena centre.
 * Exits only when fully safe for an extended period.
 * ---------------------------------------------------------- */
static void run_recovery(void)
{
  if (ctx.state_entry) {
    DEBUG_PRINT("Entering RECOVERY  pos=(%.2f, %.2f)",
                status.pos_x, status.pos_y);
    ctx.recovery_heading = compute_inward_heading();
  }

  command_forward_heading_hold(0.3f, ctx.recovery_heading);

  if (!status.out_of_bounds   &&
      !status.near_bounds      &&
      !status.obstacle_imminent &&
      ctx.bounds_safe_count >= 5) {
    ctx.preferred_heading = status.heading;
    change_state(EXPLORE);
    return;
  }
}


/* ============================================================
 * Command helpers
 * ============================================================ */

static void command_stop(void)
{
  guidance_h_set_body_vel(0.0f, 0.0f);
  guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi);
}

static void command_forward_heading_rate(float forward_speed, float heading_rate_rad_s)
{
  guidance_h_set_body_vel(forward_speed, 0.0f);
  guidance_h_set_heading_rate(heading_rate_rad_s);
}

static void command_forward_heading_hold(float forward_speed, float heading_rad)
{
  guidance_h_set_body_vel(forward_speed, 0.0f);
  guidance_h_set_heading(heading_rad);
}

static void command_hover_heading_hold(float heading_rad)
{
  guidance_h_set_body_vel(0.0f, 0.0f);
  guidance_h_set_heading(heading_rad);
}


/* ============================================================
 * Utilities
 * ============================================================ */

static bool state_timed_out(float timeout_s)
{
  const int limit = (int)(timeout_s * MISSION_PERIODIC_HZ);
  return (ctx.state_ticks >= limit);
}

/*
 * Returns the heading (radians, ENU/body frame) pointing toward (0, 0)
 * from the current position.  Used for boundary recovery.
 *
 * In ENU: x = East, y = North.
 * atan2f(dx, dy) gives the heading angle measured clockwise from North,
 * which matches Paparazzi's psi convention.
 */
static float compute_inward_heading(void)
{
  float dx = 0.0f - status.pos_x;
  float dy = 0.0f - status.pos_y;

  /* Avoid atan2(0,0) at the exact centre — keep current heading */
  if (fabsf(dx) < 0.01f && fabsf(dy) < 0.01f) {
    return status.heading;
  }

  return atan2f(dx, dy);
}

/*
 * wrap_angle — normalise angle to [-pi, pi].
 */
static float wrap_angle(float a)
{
  while (a >  M_PI) a -= 2.0f * M_PI;
  while (a < -M_PI) a += 2.0f * M_PI;
  return a;
}

/*
 * far_side_heading — returns the heading toward the half of the CyberZoo
 * that is furthest from the drone's current position, using OptiTrack ENU.
 *
 * The CyberZoo is roughly centred on (0, 0).  The "far side" is simply
 * the direction opposite to where the drone currently is relative to
 * centre — i.e. pointing from current position through (0,0) and beyond.
 *
 * This gives a gentle position-aware bias: a drone near the North wall
 * will be nudged to face South, etc., ensuring the whole arena is explored
 * without any hard-coded waypoints.
 */
static float far_side_heading(void)
{
  /* Vector from drone to arena centre */
  float dx = 0.0f - status.pos_x;
  float dy = 0.0f - status.pos_y;

  /* If already very close to centre, keep current preferred heading */
  if (fabsf(dx) < 0.1f && fabsf(dy) < 0.1f) {
    return ctx.preferred_heading;
  }

  return atan2f(dx, dy);
}