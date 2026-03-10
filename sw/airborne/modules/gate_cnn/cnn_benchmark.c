#include <stdio.h>
#include "cnn.h"
#include "cnn_benchmark.h"

#define INPUT_WIDTH  240
#define INPUT_HEIGHT 520
#define INPUT_SIZE   (INPUT_WIDTH * INPUT_HEIGHT)
#define NUM_RUNS     100

/* CNN MODULE:
 * Startup benchmark for testing whether the CNN module runs inside Paparazzi.
 * For now it uses dummy input, not live camera frames yet.
 */
void gate_cnn_benchmark_init(void)
{
    static float input[INPUT_SIZE];
    float output = 0.0f;

    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = (float)i * 0.001f;
    }

    printf("CNN MODULE: benchmark started\n");

    for (int i = 0; i < NUM_RUNS; i++) {
        cnn_forward(input, &output);
    }

    printf("CNN MODULE: benchmark finished\n");
    printf("CNN MODULE: runs = %d\n", NUM_RUNS);
    printf("CNN MODULE: final output = %f\n", output);
}