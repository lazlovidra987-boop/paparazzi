/*
 * gate_navigator_cnn.c
 * --------------------
 * Flies the Bebop through a gate using the CNN detector output.
 * Input:  gate_cnn_result.heading    [-1, 1]  normalised horizontal error
 *         gate_cnn_result.confidence [0,  1]  gate visibility probability
 *
 * State machine (mirrors gate_navigator.c structure)
 * --------------------------------------------------
 *   SEARCH   — no gate found, rotate slowly
 *   ALIGN    — gate found, yaw to centre (heading → 0)
 *   APPROACH — aligned, fly forward; re-align if drift > threshold
 *   PASS     — fly straight through for cnn_nav_pass_time_s seconds
 *
 * Key difference from gate_navigator.c
 * -------------------------------------
 *   - No pixel maths: heading comes pre-normalised from the CNN
 *   - No gate size / distance: APPROACH → PASS uses a time trigger
 *   - Uses cnn_nav_* parameter names to avoid linker conflicts
 *
 * Place in: sw/airborne/modules/gate_navigator/gate_navigator_cnn.c
 */

#include <math.h>
#include <stdio.h>

#include "autopilot.h"
#include "firmwares/rotorcraft/autopilot_guided.h"
#include "firmwares/rotorcraft/guidance/guidance_h.h"
#include "firmwares/rotorcraft/guidance/guidance_v.h"   // ADD THIS
#include "generated/airframe.h"
#include "state.h"

#include "modules/gate_navigator/gate_navigator_cnn.h"
#include "modules/gate_cnn/gate_cnn.h"   /* gate_cnn_result */


/* ── Tunable parameters ─────────────────────────────────────────────── */

#ifndef CNN_NAV_FORWARD_SPEED
#define CNN_NAV_FORWARD_SPEED 0.75f
#endif
float cnn_nav_forward_speed = CNN_NAV_FORWARD_SPEED;

#ifndef CNN_NAV_CONF_THRESH
#define CNN_NAV_CONF_THRESH 0.6f
#endif
float cnn_nav_conf_thresh = CNN_NAV_CONF_THRESH;

#ifndef CNN_NAV_ALIGN_THRESH
#define CNN_NAV_ALIGN_THRESH 0.08f
#endif
float cnn_nav_align_thresh = CNN_NAV_ALIGN_THRESH;

#ifndef CNN_NAV_SEARCH_RATE
#define CNN_NAV_SEARCH_RATE 5.0f
#endif
float cnn_nav_search_rate = CNN_NAV_SEARCH_RATE;

#ifndef CNN_NAV_YAW_GAIN
#define CNN_NAV_YAW_GAIN 15.0f
#endif
float cnn_nav_yaw_gain = CNN_NAV_YAW_GAIN;

#ifndef CNN_NAV_APPROACH_TIME_S
#define CNN_NAV_APPROACH_TIME_S 4.0f
#endif
float cnn_nav_approach_time_s = CNN_NAV_APPROACH_TIME_S;

#ifndef CNN_NAV_PASS_TIME_S
#define CNN_NAV_PASS_TIME_S 2.5f
#endif
float cnn_nav_pass_time_s = CNN_NAV_PASS_TIME_S;

#define CNN_NAV_PERIODIC_HZ 10


/* ── Debug ──────────────────────────────────────────────────────────── */
#define GATE_NAV_CNN_DEBUG
#ifdef GATE_NAV_CNN_DEBUG
#define DEBUG_PRINT(fmt, ...) \
    printf("[gate_navigator_cnn] " fmt "\n", ##__VA_ARGS__)
#else
#define DEBUG_PRINT(fmt, ...) do {} while (0)
#endif


/* ── State machine ──────────────────────────────────────────────────── */
enum cnn_nav_state { CNN_SEARCH, CNN_ALIGN, CNN_APPROACH, CNN_PASS };
static enum cnn_nav_state nav_state = CNN_SEARCH;
static int state_ticks = 0;


/* ── Init ───────────────────────────────────────────────────────────── */
void gate_navigator_cnn_init(void)
{
    nav_state   = CNN_SEARCH;
    state_ticks = 0;
    printf("[gate_navigator_cnn] Initialised\n");
    printf("[gate_navigator_cnn] conf_thresh=%.2f  align_thresh=%.2f  "
           "fwd_speed=%.2f  yaw_gain=%.2f  approach_t=%.1fs  pass_t=%.1fs\n",
           cnn_nav_conf_thresh, cnn_nav_align_thresh,
           cnn_nav_forward_speed, cnn_nav_yaw_gain,
           cnn_nav_approach_time_s, cnn_nav_pass_time_s);
}


/* ── Periodic (10 Hz) ───────────────────────────────────────────────── */
void gate_navigator_cnn_periodic(void)
{
    /* Only act in GUIDED mode */
    if (guidance_h.mode != GUIDANCE_H_MODE_GUIDED) {
        return;
    }
    guidance_v_set_vz(0.0f);
    /* ── Read CNN output ────────────────────────────────────────────── */
    float   heading    = gate_cnn_result.heading;     /* [-1, 1] */
    float   confidence = gate_cnn_result.gate_measure;  /* [0,  1] */
    uint8_t gate_found = (confidence >= cnn_nav_conf_thresh) ? 1 : 0;

    state_ticks++;

    switch (nav_state) {

        /* ── SEARCH ─────────────────────────────────────────────── */
        case CNN_SEARCH:
            DEBUG_PRINT("SEARCH  conf=%.3f  heading=%+.3f",
                        confidence, heading);

            guidance_h_set_body_vel(0.f, 0.f);
            guidance_h_set_heading_rate(RadOfDeg(cnn_nav_search_rate));

            if (gate_found) {
                DEBUG_PRINT("Gate found -> ALIGN");
                nav_state   = CNN_ALIGN;
                state_ticks = 0;
            }
            break;

        /* ── ALIGN ──────────────────────────────────────────────── */
        case CNN_ALIGN:
            DEBUG_PRINT("ALIGN   conf=%.3f  heading=%+.3f",
                        confidence, heading);

            if (!gate_found) {
                DEBUG_PRINT("Gate lost -> SEARCH");
                nav_state   = CNN_SEARCH;
                state_ticks = 0;
                break;
            }

            guidance_h_set_body_vel(0.f, 0.f);

            if (fabsf(heading) < cnn_nav_align_thresh) {
                guidance_h_set_heading_rate(0.f);
                DEBUG_PRINT("Aligned -> APPROACH");
                nav_state   = CNN_APPROACH;
                state_ticks = 0;
            } else {
                /* Proportional yaw towards gate centre */
                guidance_h_set_heading_rate(
                    RadOfDeg(heading * cnn_nav_yaw_gain));
            }
            break;

        /* ── APPROACH ───────────────────────────────────────────── */
        case CNN_APPROACH:
            DEBUG_PRINT("APPROACH  conf=%.3f  heading=%+.3f  ticks=%d",
                        confidence, heading, state_ticks);

            if (!gate_found) {
                DEBUG_PRINT("Gate lost -> SEARCH");
                guidance_h_set_body_vel(0.f, 0.f);
                guidance_h_set_heading_rate(0.f);
                nav_state   = CNN_SEARCH;
                state_ticks = 0;
                break;
            }

            /* Drifted too far off-centre: stop and re-align */
            if (fabsf(heading) > 0.20f) {
                DEBUG_PRINT("Drifted (heading=%+.3f) -> ALIGN", heading);
                guidance_h_set_body_vel(0.f, 0.f);
                nav_state   = CNN_ALIGN;
                state_ticks = 0;
                break;
            }

            /* Time-based PASS trigger (no distance info from CNN) */
            if (state_ticks >=
                    (int)(cnn_nav_approach_time_s * CNN_NAV_PERIODIC_HZ)) {
                DEBUG_PRINT("Approach time elapsed -> PASS");
                nav_state   = CNN_PASS;
                state_ticks = 0;
                break;
            }

            /* Fly forward with gentle heading correction */
            guidance_h_set_body_vel(cnn_nav_forward_speed, 0.f);
            guidance_h_set_heading_rate(
                RadOfDeg(heading * cnn_nav_yaw_gain * 0.4f));
            break;

        /* ── PASS ───────────────────────────────────────────────── */
        case CNN_PASS:
            DEBUG_PRINT("PASS  ticks=%d/%d",
                        state_ticks,
                        (int)(cnn_nav_pass_time_s * CNN_NAV_PERIODIC_HZ));

            /* Fly straight, no corrections */
            guidance_h_set_body_vel(cnn_nav_forward_speed, 0.f);
            guidance_h_set_heading_rate(0.f);

            if (state_ticks >=
                    (int)(cnn_nav_pass_time_s * CNN_NAV_PERIODIC_HZ)) {
                DEBUG_PRINT("Gate passed -> SEARCH");
                nav_state   = CNN_SEARCH;
                state_ticks = 0;
            }
            break;

        default:
            nav_state   = CNN_SEARCH;
            state_ticks = 0;
            break;
    }
}