#ifndef GATE_NAVIGATOR_H
#define GATE_NAVIGATOR_H

extern void gate_navigator_init(void);
extern void gate_navigator_periodic(void);

/* Tunable via GCS Settings */
extern float nav_forward_speed;   // m/s when approaching gate
extern float nav_quality_thresh;  // min gate quality to act on
extern float nav_align_thresh;    // normalized error to consider "aligned"
extern float nav_search_rate;     // deg/s rotation when searching

#endif