# Obstacle Safety Threshold Solution

## Problem Analysis

The current obstacle detection system has two main weaknesses:
1. **Downsampling loss**: The image is downsampled into blocks (default 4x4), causing:
   - Loss of precise edge information
   - Obstacles appearing smaller than they actually are
   - Boundary detection inaccuracies

2. **Edge detection inaccuracy**: The carpet/obstacle detection uses simple gradient filtering:
   - May miss obstacle boundaries
   - Doesn't account for detection uncertainty
   - Can create false negatives at obstacle margins

## Proposed Solutions (in order of recommendation)

### Solution 1: Morphological Dilation (RECOMMENDED - Most Effective)
**Best for**: Enlarging detected obstacles with a proper safety margin

**How it works**:
- Apply dilation to the `carpet_small[][]` and `ground_small[][]` masks
- Dilation expands detected obstacles by spreading them to neighboring cells
- Multiple passes increase the safety margin

**Benefits**:
- Geometric expansion ensures complete safety boundary
- Doesn't require tuning multiple parameters
- Works naturally with the existing downsampled grid

**Implementation**:
```c
// Add dilation function
static void dilate_map(uint8_t map[GS_MAX_ROWS][GS_MAX_COLS], 
                       uint16_t cols, uint16_t rows, 
                       uint16_t passes)
{
  for (uint16_t pass = 0U; pass < passes; pass++) {
    uint8_t temp_map[GS_MAX_ROWS][GS_MAX_COLS];
    memcpy(temp_map, map, sizeof(temp_map));
    
    for (uint16_t r = 0U; r < rows; r++) {
      for (uint16_t c = 0U; c < cols; c++) {
        if (map[r][c] != 0U) {
          // Spread to all neighbors (4-connected or 8-connected)
          if (r > 0U) temp_map[r-1U][c] = 1U;
          if (r < rows-1U) temp_map[r+1U][c] = 1U;
          if (c > 0U) temp_map[r][c-1U] = 1U;
          if (c < cols-1U) temp_map[r][c+1U] = 1U;
        }
      }
    }
    memcpy(map, temp_map, sizeof(temp_map));
  }
}
```

**Where to apply**:
- After `find_carpet()` and before combining masks
- Tuning parameter: `uint8_t ground_safety_dilation_passes = 2U;` (1-3)

---

### Solution 2: Horizon Reduction (Simple & Lightweight)
**Best for**: Quick deployment, minimal code changes

**How it works**:
- Reduce all horizon values by a safety offset
- Makes obstacles appear closer to the drone
- Triggers obstacle detection sooner

**Benefits**:
- Single parameter to tune
- Minimal overhead
- Works immediately without restructuring

**Implementation in `compute_horizon()`**:
```c
#define HORIZON_SAFETY_MARGIN 3U  // Reduce horizon by this many blocks

// In compute_horizon(), after final horizon computation:
for (uint16_t c = 0U; c < cols; c++) {
  if (res->horizon[c] > HORIZON_SAFETY_MARGIN) {
    res->horizon[c] -= HORIZON_SAFETY_MARGIN;
  } else if (res->horizon[c] > 0U) {
    res->horizon[c] = 1U;  // At least signal something is there
  }
}
```

**Tuning**: `HORIZON_SAFETY_MARGIN = 2-4` (scales with `ground_downsize_x/y`)

---

### Solution 3: Temporal Filtering (Prevents False Positives)
**Best for**: Eliminating transient noise and false detections

**How it works**:
- Require obstacle detection to persist across multiple frames
- Maintain frame history of obstacle state
- Only signal obstacle if consistent over N frames

**Benefits**:
- Rejects single-frame noise
- Reduces false obstacle warnings
- Makes navigation more stable

**Implementation**:
```c
#define OBSTACLE_PERSISTENCE_FRAMES 3U

static uint8_t obstacle_frame_counter = 0U;
static bool obstacle_reported = false;

// In compute_obstacle_flag():
void compute_obstacle_flag(struct ground_seg_result_t *res)
{
  res->obstacle_ahead = false;
  
  // ... existing computation ...
  bool obstacle_detected = /* computed result */;
  
  if (obstacle_detected) {
    if (obstacle_frame_counter < 255U) {
      obstacle_frame_counter++;
    }
  } else {
    obstacle_frame_counter = 0U;
  }
  
  // Only report if persistent
  if (obstacle_frame_counter >= OBSTACLE_PERSISTENCE_FRAMES) {
    res->obstacle_ahead = true;
    obstacle_reported = true;
  } else if (obstacle_reported && obstacle_frame_counter >= OBSTACLE_PERSISTENCE_FRAMES / 2U) {
    // Hysteresis: keep reporting until evidence clearly disappears
    res->obstacle_ahead = true;
  } else {
    obstacle_reported = false;
  }
}
```

---

### Solution 4: Blocked Column Threshold (Confidence-Based)
**Best for**: Requiring multiple affected columns before triggering

**How it works**:
- Count how many center columns are actually blocked
- Require minimum percentage (e.g., 50%) to report obstacle
- Prevents triggering on small gaps

**Benefits**:
- More robust to partial occlusions
- Geometric consistency check
- Parameter easily tunable

**Implementation**:
```c
#define MIN_BLOCKED_COLUMNS_PCT 50U  // Require 50% of middle cols blocked

void compute_obstacle_flag(struct ground_seg_result_t *res)
{
  res->obstacle_ahead = false;
  
  // ... existing setup ...
  
  uint16_t blocked_cols = 0U;
  
  for (uint16_t c = start; c < end; c++) {
    if (res->horizon[c] <= 1U) {  // Column is blocked/very short
      blocked_cols++;
    }
  }
  
  uint16_t blocked_pct = (blocked_cols * 100U) / total;
  
  if (blocked_pct >= MIN_BLOCKED_COLUMNS_PCT) {
    res->obstacle_ahead = true;
  }
}
```

---

### Solution 5: Gradient-Based Dilation (For Carpet Detection)
**Best for**: Expanding carpet masks before combining with ground map

**How it works**:
- Apply dilation specifically to carpet_small[][] before combining
- Ensures carpet edges have safety margin
- Works in tandem with Solution 1

**Implementation**:
```c
// In ground_seg_analyse_image(), after find_carpet():
dilate_map(carpet_small, cols, rows, 2U);  // 2 passes of dilation

// Then combine
for (uint16_t r = 0U; r < rows; r++) {
  for (uint16_t c = 0U; c < cols; c++) {
    ground_small[r][c] = ground_small[r][c] | carpet_small[r][c];
  }
}
```

---

## Recommended Implementation Strategy

### Phase 1: Immediate Safety (Start Here)
Combine **Solutions 1 + 4**:
```
1. Add morphological dilation to carpet_small (2 passes)
2. Increase MIN_BLOCKED_COLUMNS_PCT from 33% to 50%
3. Test with real obstacle scenarios
```

### Phase 2: Enhanced Robustness
Add **Solution 3** (temporal filtering):
```
Require 2-3 consecutive frames of obstacle detection
Prevents phantom obstacles from transient detection errors
```

### Phase 3: Fine-Tuning
Use **Solution 2** (horizon margin) for additional safety:
```
Reduce horizon by 2-3 blocks as final safety net
Fine-grained control per configuration
```

---

## Parameter Tuning Guide

| Parameter | Purpose | Safe Range | Notes |
|-----------|---------|-----------|-------|
| `ground_safety_dilation_passes` | Dilation iterations | 1-3 | 1 = slight expansion, 3 = large margin |
| `HORIZON_SAFETY_MARGIN` | Horizon reduction blocks | 2-5 | 1 block ≈ ground_downsize_x pixels |
| `OBSTACLE_PERSISTENCE_FRAMES` | Temporal filter | 2-5 | Higher = more stable but slower detection |
| `MIN_BLOCKED_COLUMNS_PCT` | Blocked column threshold | 40-60% | 50% is good compromise |

---

## Testing Recommendations

1. **Controlled Obstacle Test**: Move drone toward known obstacle
   - Verify obstacle_ahead triggers at safe distance
   - Observe margin consistency

2. **False Positive Test**: Reflect light, sudden shadows
   - Verify no spurious obstacles detected
   - Check temporal filter effectiveness

3. **Edge Case Test**: Narrow passages, low obstacles
   - Ensure drone can navigate confirmed passable areas
   - Safety margin should not be excessive

4. **Performance Test**: Monitor CPU usage with dilation
   - Each dilation pass ~O(cols × rows)
   - Should be negligible at typical resolutions (40-80 cols)

---

## Code Location Modifications

### cv_ground_seg.c changes:
- Add `ground_safety_dilation_passes` tunable parameter
- Add `dilate_map()` helper function
- Call dilation after `find_carpet()` 
- Modify `compute_obstacle_flag()` for blocked column check
- Add temporal filter state

### ground_seg_nav.c changes:
- Optional: Adjust `gsn_floor_frac` and `gsn_obstacle_frac` downward
- Test navigation state machine with new safety margins

