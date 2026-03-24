/*
 * IMPLEMENTATION EXAMPLE: Obstacle Safety Threshold
 * 
 * This file shows concrete code modifications for cv_ground_seg.c
 * to implement morphological dilation + temporal filtering
 */

/* ===== ADD TO cv_ground_seg.c HEADER SECTION ===== */

/*
 * Safety parameters - these replace or supplement existing tuning params
 */
uint8_t ground_safety_dilation_passes = 2U;   /* 1-3: expand obstacles by this many cells */
uint8_t ground_obstacle_min_blocked_pct = 50U; /* Percent of center columns that must be blocked */

/*
 * Temporal filtering state
 */
static uint8_t obstacle_confidence_counter = 0U;
static const uint8_t OBSTACLE_CONFIDENCE_THRESHOLD = 2U; /* frames to confirm */
static bool obstacle_last_reported = false;


/* ===== ADD HELPER FUNCTION ===== */

/*
 * Morphological dilation - expands obstacles by 1 cell per iteration
 * 
 * This smooths the binary map and fills small holes, creating a safety buffer.
 * 4-connected neighborhood (no diagonals) to avoid over-expansion.
 */
static void dilate_obstacle_map(uint8_t map[GS_MAX_ROWS][GS_MAX_COLS],
                                uint16_t cols, uint16_t rows,
                                uint16_t passes)
{
  uint8_t temp_map[GS_MAX_ROWS][GS_MAX_COLS];

  for (uint16_t pass = 0U; pass < passes; pass++) {
    /* Copy current state */
    memcpy(temp_map, map, cols * rows * sizeof(uint8_t));

    /* For each cell, if it's obstacle, mark all neighbors as obstacle */
    for (uint16_t r = 0U; r < rows; r++) {
      for (uint16_t c = 0U; c < cols; c++) {
        if (map[r][c] != 0U) {
          /* Mark cell itself (already done) */
          temp_map[r][c] = 1U;

          /* Mark 4-connected neighbors */
          if (r > 0U) {
            temp_map[r - 1U][c] = 1U;
          }
          if (r < rows - 1U) {
            temp_map[r + 1U][c] = 1U;
          }
          if (c > 0U) {
            temp_map[r][c - 1U] = 1U;
          }
          if (c < cols - 1U) {
            temp_map[r][c + 1U] = 1U;
          }
        }
      }
    }

    /* Copy dilated result back */
    memcpy(map, temp_map, cols * rows * sizeof(uint8_t));
  }
}


/* ===== MODIFICATION 1: In ground_seg_analyse_image() ===== */

static uint32_t ground_seg_analyse_image(struct image_t *img,
                                         struct ground_seg_result_t *res)
{
  reset_result(res);

  if (img == NULL || img->buf == NULL) {
    return 0U;
  }

  if (ground_downsize_x == 0U || ground_downsize_y == 0U) {
    return 0U;
  }

  uint16_t lw = logical_width(img);
  uint16_t lh = logical_height(img);

  uint16_t cols = lw / ground_downsize_x;
  uint16_t rows = lh / ground_downsize_y;

  if (cols == 0U || rows == 0U) {
    return 0U;
  }

  if (cols > GS_MAX_COLS) {
    cols = GS_MAX_COLS;
  }
  if (rows > GS_MAX_ROWS) {
    rows = GS_MAX_ROWS;
  }

  res->cols = cols;
  res->rows = rows;

  /* Create carpet mask */
  find_carpet(img, cols, rows);

  debug_print_image_info(img, cols, rows);
  debug_print_raw_samples(img);

  /* ===== NEW SAFETY CODE: Apply dilation to carpet map ===== */
  if (ground_safety_dilation_passes > 0U) {
    dilate_obstacle_map(carpet_small, cols, rows, ground_safety_dilation_passes);
    GS_PRINT("Dilated carpet map with %u passes\n", ground_safety_dilation_passes);
  }

  build_ground_map(img, cols, rows);
  debug_print_map_counts(cols, rows);

  /* Combining the masks - carpet now has expanded safety margin */
  for (uint16_t r = 0U; r < rows; r++) {
    for (uint16_t c = 0U; c < cols; c++) {
      ground_small[r][c] = ground_small[r][c] | carpet_small[r][c];
    }
  }

  compute_horizon(res);
  debug_print_horizon_samples(res);

  compute_scores(res);
  compute_obstacle_flag(res);              /* MODIFIED: see below */
  compute_centroid_and_ground_count(res);

  return res->ground_count;
}


/* ===== MODIFICATION 2: Replace compute_obstacle_flag() ===== */

/*
 * Enhanced obstacle detection with:
 * 1. Blocked column percentage threshold
 * 2. Temporal filtering for stability
 * 3. Hysteresis to prevent chatter
 */
static void compute_obstacle_flag(struct ground_seg_result_t *res)
{
  res->obstacle_ahead = false;

  if (res->cols == 0U) {
    obstacle_confidence_counter = 0U;
    obstacle_last_reported = false;
    return;
  }

  uint16_t middle_cols = ground_middle_cols;
  if (middle_cols == 0U) {
    middle_cols = 1U;
  }
  if (middle_cols > res->cols) {
    middle_cols = res->cols;
  }

  uint16_t mid = res->cols / 2U;
  uint16_t half = middle_cols / 2U;

  uint16_t start = (mid > half) ? (uint16_t)(mid - half) : 0U;
  uint16_t end   = start + middle_cols;
  if (end > res->cols) {
    end = res->cols;
  }

  /* --- NEW: Count blocked vs open columns --- */
  uint16_t blocked_cols = 0U;
  uint16_t total_cols = 0U;

  for (uint16_t c = start; c < end; c++) {
    total_cols++;
    /* A column is "blocked" if horizon is very small (close obstacle) */
    if (res->horizon[c] <= 1U) {
      blocked_cols++;
    }
  }

  /* --- NEW: Check percentage threshold --- */
  uint16_t blocked_pct = (total_cols > 0U) ? ((blocked_cols * 100U) / total_cols) : 0U;

  bool raw_obstacle_detected = (blocked_pct >= ground_obstacle_min_blocked_pct);

  if (should_print_debug()) {
    GS_PRINT("OBSTACLE CHECK | blocked=%u/%u (%.0f%%) threshold=%u%% | raw=%d\n",
             blocked_cols, total_cols, 
             (float)blocked_pct,
             ground_obstacle_min_blocked_pct,
             raw_obstacle_detected);
  }

  /* --- NEW: Apply temporal filtering --- */
  if (raw_obstacle_detected) {
    /* Raw detection is true, increment confidence */
    if (obstacle_confidence_counter < 255U) {
      obstacle_confidence_counter++;
    }
  } else {
    /* Raw detection is false, decrement confidence */
    if (obstacle_confidence_counter > 0U) {
      obstacle_confidence_counter--;
    } else {
      obstacle_last_reported = false;
    }
  }

  /* --- NEW: Hysteresis on confidence threshold --- */
  /* 
   * Turn ON when confidence >= threshold
   * Stay ON until confidence drops below threshold/2 (hysteresis)
   * This prevents chattering on threshold boundary
   */
  if (obstacle_confidence_counter >= OBSTACLE_CONFIDENCE_THRESHOLD) {
    res->obstacle_ahead = true;
    obstacle_last_reported = true;
  } else if (obstacle_last_reported && obstacle_confidence_counter > 0U) {
    res->obstacle_ahead = true;  /* Hysteresis: stay ON briefly */
  } else {
    res->obstacle_ahead = false;
    obstacle_last_reported = false;
  }

  if (should_print_debug()) {
    GS_PRINT("TEMPORAL FILTER | confidence=%u threshold=%u | reported=%d\n",
             obstacle_confidence_counter,
             OBSTACLE_CONFIDENCE_THRESHOLD,
             res->obstacle_ahead);
  }
}


/* ===== OPTIONAL MODIFICATION 3: Horizon Safety Margin ===== */

/*
 * If you want additional safety: reduce all horizon values after computation
 * This makes obstacles appear closer to the drone
 * Apply in compute_centroid_and_ground_count() BEFORE computing centroid
 */

/* Add to top of file: */
#define HORIZON_SAFETY_MARGIN 2U  /* blocks to subtract from horizon */

/* In compute_centroid_and_ground_count(), after compute_horizon() returns: */
static void compute_centroid_and_ground_count(struct ground_seg_result_t *res)
{
  /* ===== NEW: Apply horizon safety margin ===== */
  if (HORIZON_SAFETY_MARGIN > 0U) {
    for (uint16_t c = 0U; c < res->cols; c++) {
      if (res->horizon[c] > HORIZON_SAFETY_MARGIN) {
        res->horizon[c] -= HORIZON_SAFETY_MARGIN;
      } else if (res->horizon[c] > 0U) {
        res->horizon[c] = 1U;  /* Keep at least 1 to indicate presence */
      }
    }

    if (should_print_debug()) {
      GS_PRINT("Applied horizon safety margin: -%u blocks\n", HORIZON_SAFETY_MARGIN);
    }
  }

  /* Original centroid computation follows... */
  uint32_t total = 0U;
  uint32_t sum_x = 0U;
  uint32_t sum_y = 0U;

  for (uint16_t c = 0U; c < res->cols; c++) {
    uint16_t h = res->horizon[c];
    if (h == 0U) {
      continue;
    }

    total += h;
    sum_x += (uint32_t)c * h;

    uint16_t top_row = (uint16_t)(res->rows - h);
    uint16_t bottom_row = (uint16_t)(res->rows - 1U);
    uint16_t avg_row = (uint16_t)((top_row + bottom_row) / 2U);

    sum_y += (uint32_t)avg_row * h;
  }

  res->ground_count = total;

  if (total > 0U) {
    float cx = sum_x / (float)total;
    float cy = sum_y / (float)total;

    res->x_c = (int32_t)roundf(cx - (res->cols * 0.5f));
    res->y_c = (int32_t)roundf((res->rows * 0.5f) - cy);
  } else {
    res->x_c = 0;
    res->y_c = 0;
  }
}


/* ===== RESET FUNCTION FOR SAFE STATE TRANSITIONS ===== */

/*
 * Call this from ground_segmentation_init() to reset temporal filter state
 */
static void reset_temporal_filter(void)
{
  obstacle_confidence_counter = 0U;
  obstacle_last_reported = false;
  GS_PRINT("Temporal filter state reset\n");
}

/* Add to ground_segmentation_init(): */
void ground_segmentation_init(void)
{
  memset(&ground_seg_shared, 0, sizeof(ground_seg_shared));
  pthread_mutex_init(&ground_seg_mutex, NULL);
  gs_frame_counter = 0U;
  
  /* NEW: Reset temporal filter */
  reset_temporal_filter();

#ifdef GROUND_SEGMENTATION_DRAW
  ground_draw = GROUND_SEGMENTATION_DRAW;
#endif

#ifdef GROUND_SEGMENTATION_CAMERA
  cv_add_to_device(&GROUND_SEGMENTATION_CAMERA,
                   ground_seg_process_image,
                   GROUND_SEGMENTATION_FPS,
                   0);
#endif

  GS_PRINT("Ground segmentation initialized\n");
}


/* ===== SETTINGS CONFIGURATION ===== */

/*
 * Add these to your airframe's settings XML to make parameters tunable:

<section name="GROUND_SEGMENTATION" prefix="GROUND_">
  <!-- Safety parameters -->
  <define name="SAFETY_DILATION_PASSES" value="2" type="int" unit="passes"/>
  <define name="OBSTACLE_MIN_BLOCKED_PCT" value="50" type="int" unit="%"/>
  <define name="HORIZON_SAFETY_MARGIN" value="2" type="int" unit="blocks"/>
  
  <!-- Original parameters still available -->
  <define name="DOWNSIZE_X" value="4" type="int" unit="blocks"/>
  <define name="DOWNSIZE_Y" value="4" type="int" unit="blocks"/>
  <define name="MIN_BLACK" value="5" type="int" unit="blocks"/>
  <define name="MIDDLE_COLS" value="10" type="int"/>
</section>

 */

