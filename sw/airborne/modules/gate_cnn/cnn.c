#include "cnn.h"
#include "model_weights.h"

/* CNN MODULE:
 * Minimal forward pass for initial Paparazzi integration test.
 * This is NOT the final CNN yet.
 *
 * Right now this function is intentionally simple.
 * Later, replace the body of this function with the real CNN inference.
 */
void cnn_forward(const float *input, float *output)
{
    float sum = 0.0f;

    /* CNN MODULE: dummy workload */
    for (int i = 0; i < 100; i++) {
        sum += input[i];
    }

    /* CNN MODULE: verify weights are linked correctly */
    sum += features_0_weight[0];

    *output = sum;
}