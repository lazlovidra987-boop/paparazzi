# Visual Comparison: Before vs After Safety Thresholds

## Problem Scenario

### BEFORE: Inaccurate Obstacle Detection
```
Physical Reality            Downsampled Binary Map      Detected Horizon   
(Camera View)               (4x4 blocks)                per column
                                                        
  [  OBSTACLE  ]               [#][#][#]                #[ 0][ 0][ 0][#]
  [  OBSTACLE  ]               [#][#][#]          →     #[ 2][ 3][ 1][#]  ← Not accurate!
  [GROUND][....][GROUND]  →    [G][#][G]                #[10][12][11][#]
  [GROUND][....][GROUND]       [G][#][G]                #[20][22][21][#]
  
Result: Obstacle appears only 1-3 blocks away, drone too close
Risk: Drone might move into obstacle before reacting
```

---

## Solution 1: Morphological Dilation
### Example with 2 dilation passes

**Before Dilation:**
```
Carpet Map (detected edges)     Ground Map           Combined
[.][.][.][.][.]                [G][G][G][G][G]      [G][G][G][G][G]
[.][#][.][.][.]                [G][#][G][G][G]  →   [G][#][G][G][G]
[.][#][.][.][.]                [G][#][G][G][G]      [G][#][G][G][G]
[.][.][.][.][.]                [G][G][G][G][G]      [G][G][G][G][G]
```

**After 1st Dilation Pass:**
```
Carpet map expanded by 1 cell
[.][.][.][.][.]
[#][#][#][.][.]  ← Obstacle spreads horizontally/vertically
[#][#][#][.][.]
[.][#][.][.][.]
```

**After 2nd Dilation Pass:**
```
Carpet map expanded further
[.][#][.][.][.]
[#][#][#][#][.]  ← Creates safety buffer around obstacle
[#][#][#][#][.]
[#][#][#][.][.]
```

**Result:**
```
Obstacle boundary is now clearly marked with safety margin
Horizon computation treats expanded obstacle as closer
Drone maintains safe distance at all times
```

---

## Solution 2: Blocked Column Percentage Threshold

### Example: Requiring 50% of center columns to be blocked

**Scenario 1: Partial Obstruction (transient sensor error)**
```
Center region (columns 3,4,5)
horizon: [5][2][6]  ← One column shows small horizon

Blocked count: 1 out of 3 = 33%
Threshold: 50%
Result: SAFE - Not a real obstacle, likely false reading
```

**Scenario 2: Real Obstacle Ahead**
```
Center region (columns 3,4,5)
horizon: [1][0][2]  ← Multiple columns show very small horizon

Blocked count: 2-3 out of 3 = 67-100%
Threshold: 50%
Result: BLOCKED - Real obstacle detected
Drone triggers avoidance
```

**Grid-level view:**
```
Real obstacle:          Noise/false reading:
[OBSTACLE][...]        [noise][sky]
[OBSTACLE][...]   vs   [...][ground]
[.......][GROUND]      [GROUND][GROUND]

Horizon: [0][0]        Horizon: [3][15]
Blocked: 100%          Blocked: 0%
```

---

## Solution 3: Temporal Filtering (Confidence Counter)

### Frame-by-frame behavior with OBSTACLE_CONFIDENCE_THRESHOLD = 2

**Scenario: Real obstacle approaching**
```
Frame 1  Frame 2  Frame 3  Frame 4  Frame 5
Raw:   T    T      T       F        F
Conf:  1    2      3       2        1
Out:   F    T      T       T        F
       ↑    ↑      ↑       ↑        ↑
    Hold  Trigger Report  Hysteresis Fade
              |________|_________|
           Prevents spurious alerts
```

**Scenario: Noise (single-frame false detection)**
```
Frame 1  Frame 2  Frame 3
Raw:   T    F      F      ← True for just 1 frame
Conf:  1    0      0
Out:   F    F      F      ← Never triggers!
       ↑
    Counter never reaches threshold
    Noise filtered out safely
```

**Effect:** Eliminates detection jitter while keeping responsiveness

---

## Solution 4: Horizon Safety Margin

### Reducing all horizon values by 2 blocks

**Before:**
```
Column:     1      2      3      4      5
horizon:  [20]   [21]   [19]   [22]   [18]  ← Raw horizon
Obstacle
danger:    SAFE   SAFE   SAFE   SAFE   SAFE   (far away)
```

**After applying -2 block margin:**
```
Column:     1      2      3      4      5
horizon:  [18]   [19]   [17]   [20]   [16]  ← Reduced
Obstacle
danger:   CLOSER CLOSER CLOSER CLOSER CLOSER  (safety buffer added)

Real distance: 20 blocks
Perceived as:  18 blocks
Safety margin: 2 blocks = 8-16 pixels @ 4x4 downsampling
```

---

## Combined Effect: All Solutions Together

### Example scenario with vs without safety measures

**BEFORE (Original Code):**
```
Obstacle at real distance: 25 pixels ahead

Frame-by-frame detection:
F1: Raw reads 20px  → sensor noise F2: Raw reads 18px  → noise spike
F3: Real obstacle 15px → triggers obstacle_ahead? Maybe...
F4: Noise 22px  → obstacle cancels??
F5: Real 12px  → LATE warning!

Drone at 8px from collision - DANGEROUS!
```

**AFTER (All Safety Solutions):**
```
Obstacle at real distance: 25 pixels ahead

F1: Dilation spreads carpet mask
    Blocked check: 40% → not enough (need 50%)
    Temporal: conf=0
    Result: SAFE

F2: Raw reads noise
    Blocked check: 30%
    Temporal: conf=0 (noise rejected)
    Result: SAFE
    
F3: Real obstacle detected
    Dilation makes it 2 blocks larger (8px safety)
    Blocked check: 60% ✓
    Temporal: conf=1
    Result: CAUTION (counter rising)
    
F4: Still detected
    Temporal: conf=2 ✓ (threshold reached!)
    Result: BLOCKED - Turn to safety NOW!

Drone turns at 18px away - SAFE!
Margin: 18px > 12px confidence distance
```

---

## Parameter Tuning Impact

### Dilation Passes Effect
```
Obstacle size in real image: ~40 pixels (5 blocks)

0 passes:  Detected as: ▓▓▓ (3 blocks)     ← Unsafe
1 pass:    Detected as: ▓▓▓▓▓ (5 blocks)   ← Better
2 passes:  Detected as: ▓▓▓▓▓▓▓ (7 blocks)  ← Preferred
3 passes:  Detected as: ▓▓▓▓▓▓▓▓▓ (9 blocks) ← Conservative
```

### Blocked Column Threshold Effect
```
Obstacle blocks 3 of 5 center columns (60%)

30%: Many false positives ← Unsafe
40%: Some false positives  
50%: Good balance ← RECOMMENDED
60%: Might miss partial blocks
70%: Missed obstacle risk ← Unsafe
```

### Horizon Margin Effect
```
Real obstacle distance: 20 blocks away

0 margin:  Treated as 20 blocks ← Original
1 margin:  Treated as 19 blocks
2 margin:  Treated as 18 blocks ← Adds 4-8 pixel safety
3 margin:  Treated as 17 blocks
```

### Confidence Threshold Effect
```
Frames needed to confirm obstacle

1: Responds immediately but noisy
2: Good balance, filters noise ← RECOMMENDED  
3: Slower response, very stable
4: Undesirable lag
```

---

## Testing Scenarios

### Test 1: Known Obstacle Avoidance
**Setup:** Place drone 50cm from wall
**Expected:** Obstacle detected, drone stops before collision
**Measurement:** Safety margin achieved?

```
[WALL]
  ← 50cm
     [DETECTABLE SAFETY BOUNDARY]
        ← 30cm
           [ACTUAL DRONE POSITION]
           
Target: Detect BEFORE drone reaches 30cm
```

### Test 2: False Positive Rejection
**Setup:** Sudden light reflection, shadow change
**Expected:** Single-frame noise ignored
**Measurement:** No spurious obstacle_ahead signals

```
Frame 1: Sensor misfire → conf=1 → SAFE
Frame 2: Normal reading → conf=0 → SAFE
(Noise never accumulates)
```

### Test 3: Dynamic Obstacle Approach
**Setup:** Slow approach to obstacle
**Expected:** Smooth detection ramp
**Measurement:** Temporal confidence counter shows steady rise

```
Frames:  1  2  3  4  5  6
Conf:    0  0  1  2  2  2
Status:  S  S  C  B  B  B
         ↑  ↑  ↑  ↑
      Safe Caution Block
```

### Test 4: Narrow Passage Navigation
**Setup:** Passage which is navigable but close
**Expected:** Successfully navigate without triggering
**Measurement:** Blocked_pct stays below threshold

```
[WALL]
     ╱─────────╲  ← 60cm wide passage
    │ FREE ZONE │
     ╲─────────╱
[WALL]

Horizon: [15][18][16][17][15]
Blocked: 0% → Can safely proceed
```

