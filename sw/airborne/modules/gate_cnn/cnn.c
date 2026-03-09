#include "cnn.h"
#include "model_weights.h"

/* CNN MODULE:
 * Minimal forward pass for initial Paparazzi integration test.
 * This is NOT the final CNN yet.
 */
void cnn_forward(const float *input, float *output)
{
    float sum = 0.0f;

    for (int i = 0; i < 100; i++) {
        sum += input[i];
    }

    sum += features_0_weight[0];

    *output = sum;
}