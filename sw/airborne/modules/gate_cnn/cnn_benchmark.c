#include <stdio.h>
#include <time.h>
#include <float.h>

#include "cnn.h"
#include "cnn_benchmark.h"

#define INPUT_HEIGHT  520
#define INPUT_WIDTH   240
#define INPUT_SIZE    (INPUT_HEIGHT * INPUT_WIDTH)

/* CNN MODULE:
 * Number of inferences done inside one periodic call.
 * This improves timing accuracy when one inference is very fast.
 */
#define BENCH_REPEAT_PER_TICK  20

/* CNN MODULE:
 * Print statistics every N periodic calls.
 * If your XML periodic frequency is 10 Hz and REPORT_EVERY_TICKS = 20,
 * you get one report every ~2 seconds.
 */
#define REPORT_EVERY_TICKS     20

/* CNN MODULE: persistent dummy input buffer */
static float benchmark_input[INPUT_SIZE];

/* CNN MODULE: benchmark statistics */
static unsigned long benchmark_tick_count = 0;
static unsigned long benchmark_total_inferences = 0;

static double benchmark_sum_ms = 0.0;
static double benchmark_min_ms = DBL_MAX;
static double benchmark_max_ms = 0.0;
static double benchmark_last_ms = 0.0;

static float benchmark_last_output = 0.0f;

/* CNN MODULE:
 * Monotonic timer in milliseconds.
 * This works well in NPS and on Linux-based Bebop targets.
 */
static double gate_cnn_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (double)ts.tv_sec * 1000.0 +
           (double)ts.tv_nsec / 1000000.0;
}

void gate_cnn_benchmark_init(void)
{
    /* CNN MODULE: create deterministic dummy input */
    for (int i = 0; i < INPUT_SIZE; i++) {
        benchmark_input[i] = (float)i * 0.001f;
    }

    /* CNN MODULE: reset statistics */
    benchmark_tick_count = 0;
    benchmark_total_inferences = 0;
    benchmark_sum_ms = 0.0;
    benchmark_min_ms = DBL_MAX;
    benchmark_max_ms = 0.0;
    benchmark_last_ms = 0.0;
    benchmark_last_output = 0.0f;

    /* CNN MODULE: optional warmup runs */
    for (int i = 0; i < 5; i++) {
        cnn_forward(benchmark_input, &benchmark_last_output);
    }

    printf("CNN MODULE: periodic benchmark initialized\n");
    printf("CNN MODULE: input size = %d floats\n", INPUT_SIZE);
    printf("CNN MODULE: repeat per tick = %d\n", BENCH_REPEAT_PER_TICK);
    printf("CNN MODULE: waiting for periodic benchmark reports...\n");
}

void gate_cnn_benchmark_periodic(void)
{
    double t0 = gate_cnn_now_ms();

    for (int i = 0; i < BENCH_REPEAT_PER_TICK; i++) {
        cnn_forward(benchmark_input, &benchmark_last_output);
    }

    double t1 = gate_cnn_now_ms();

    /* Total time for this batch */
    double batch_ms = t1 - t0;

    /* Average time per inference */
    double per_inference_ms = batch_ms / (double)BENCH_REPEAT_PER_TICK;

    benchmark_last_ms = per_inference_ms;
    benchmark_sum_ms += per_inference_ms;

    if (per_inference_ms < benchmark_min_ms) {
        benchmark_min_ms = per_inference_ms;
    }
    if (per_inference_ms > benchmark_max_ms) {
        benchmark_max_ms = per_inference_ms;
    }

    benchmark_tick_count++;
    benchmark_total_inferences += BENCH_REPEAT_PER_TICK;

    if ((benchmark_tick_count % REPORT_EVERY_TICKS) == 0) {
        double avg_ms = benchmark_sum_ms / (double)benchmark_tick_count;
        double est_max_hz = 0.0;

        if (avg_ms > 0.0) {
            est_max_hz = 1000.0 / avg_ms;
        }

        printf("CNN MODULE: ticks=%lu, total_inferences=%lu\n",
               benchmark_tick_count, benchmark_total_inferences);
        printf("CNN MODULE: inference time [ms] -> last=%.4f avg=%.4f min=%.4f max=%.4f\n",
               benchmark_last_ms, avg_ms, benchmark_min_ms, benchmark_max_ms);
        printf("CNN MODULE: estimated max raw CNN rate = %.2f Hz\n", est_max_hz);
        printf("CNN MODULE: last output = %f\n", benchmark_last_output);
    }
}