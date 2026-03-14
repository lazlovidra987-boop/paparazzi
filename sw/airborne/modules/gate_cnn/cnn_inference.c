#include "cnn_inference.h"
#include "model_weights.h"
#include <math.h>

/* ── Shape defines (DirectionGateNet) ────────────────────────────────────── */
#define IN_H    120
#define IN_W    160

/* Conv Outputs (H/W reduced by stride=2) */
#define STEM_H  60
#define STEM_W  80
#define B1_H    30
#define B1_W    40
#define B2_H    15
#define B2_W    20
#define B3_H    8
#define B3_W    10

/* Head */
#define HP_W    10
#define FC1_IN  (24 * HP_W)   /* 240 */
#define FC1_OUT 32
#define FC2_OUT 2

/* Buffers: Max size is stem output (8 * 60 * 80 = 38,400) */
#define BUF_SIZE 40000
static float buf_a[BUF_SIZE];
static float buf_b[BUF_SIZE];
static float fc1_out[FC1_OUT];
static float fc2_out[FC2_OUT];

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static void relu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) if (x[i] < 0.0f) x[i] = 0.0f;
}

/* ── Flexible Conv2d (Supports Stride and Kernel Size) ───────────────────── */

static void conv2d(
    const float *input, int in_c, int in_h, int in_w,
    const float *weight, const float *bias,
    int out_c, int k_size, int stride,
    float *output) 
{
    int out_h = in_h / stride;
    int out_w = in_w / stride;
    int pad = k_size / 2;

    for (int oc = 0; oc < out_c; oc++) {
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                float sum = bias[oc];
                int in_y_origin = y * stride - pad;
                int in_x_origin = x * stride - pad;

                for (int ic = 0; ic < in_c; ic++) {
                    for (int ky = 0; ky < k_size; ky++) {
                        int iy = in_y_origin + ky;
                        if (iy < 0 || iy >= in_h) continue;
                        for (int kx = 0; kx < k_size; kx++) {
                            int ix = in_x_origin + kx;
                            if (ix < 0 || ix >= in_w) continue;

                            sum += input[(ic * in_h + iy) * in_w + ix] * weight[((oc * in_c + ic) * k_size + ky) * k_size + kx];
                        }
                    }
                }
                output[(oc * out_h + y) * out_w + x] = sum;
            }
        }
    }
}

/* ── Adaptive Average Pool (To 1x10) ────────────────────────────────────── */

static void adaptive_avg_pool_1x10(const float *input, int c, int in_h, int in_w, float *output) {
    const float inv_h = 1.0f / (float)in_h;
    // in_w is 10, out_w is 10, so we just average the height
    for (int ch = 0; ch < c; ch++) {
        for (int x = 0; x < in_w; x++) {
            float sum = 0.0f;
            for (int y = 0; y < in_h; y++) {
                sum += input[(ch * in_h + y) * in_w + x];
            }
            output[ch * in_w + x] = sum * inv_h;
        }
    }
}

static void linear(const float *input, int in_f, const float *w, const float *b, int out_f, float *out) {
    for (int o = 0; o < out_f; o++) {
        float sum = b[o];
        for (int i = 0; i < in_f; i++) sum += w[o * in_f + i] * input[i];
        out[o] = sum;
    }
}

/* ── Main Inference ──────────────────────────────────────────────────────── */

void cnn_run(const float *image, float *heading, float *gate_measure) {
    // 1. Stem: Conv 5x5, Stride 2. [1, 120, 160] -> [8, 60, 80]
    conv2d(image, 1, 120, 160, stem_0_weight, stem_0_bias, 8, 5, 2, buf_a);
    relu_inplace(buf_a, 8 * 60 * 80);

    // 2. Block 1: Conv 3x3 (S2) -> ReLU -> Conv 3x3 (S1) -> ReLU. 
    // [8, 60, 80] -> [8, 30, 40]
    conv2d(buf_a, 8, 60, 80, block1_block_0_weight, block1_block_0_bias, 8, 3, 2, buf_b);
    relu_inplace(buf_b, 8 * 30 * 40);
    conv2d(buf_b, 8, 30, 40, block1_block_2_weight, block1_block_2_bias, 8, 3, 1, buf_a);
    relu_inplace(buf_a, 8 * 30 * 40);

    // 3. Block 2: [8, 30, 40] -> [16, 15, 20]
    conv2d(buf_a, 8, 30, 40, block2_block_0_weight, block2_block_0_bias, 16, 3, 2, buf_b);
    relu_inplace(buf_b, 16 * 15 * 20);
    conv2d(buf_b, 16, 15, 20, block2_block_2_weight, block2_block_2_bias, 16, 3, 1, buf_a);
    relu_inplace(buf_a, 16 * 15 * 20);

    // 4. Block 3: [16, 15, 20] -> [24, 8, 10]
    conv2d(buf_a, 16, 15, 20, block3_block_0_weight, block3_block_0_bias, 24, 3, 2, buf_b);
    relu_inplace(buf_b, 24 * 8 * 10);
    conv2d(buf_b, 24, 8, 10, block3_block_2_weight, block3_block_2_bias, 24, 3, 1, buf_a);
    relu_inplace(buf_a, 24 * 8 * 10);

    // 5. Horiz Pool: [24, 8, 10] -> [240 elements]
    float pooled[FC1_IN];
    adaptive_avg_pool_1x10(buf_a, 24, 8, 10, pooled);

    // 6. Head: Linear -> ReLU -> Linear
    linear(pooled, FC1_IN, head_1_weight, head_1_bias, FC1_OUT, fc1_out);
    relu_inplace(fc1_out, FC1_OUT);
    linear(fc1_out, FC1_OUT, head_3_weight, head_3_bias, FC2_OUT, fc2_out);

    // 7. Output activations
    // fc2_out[0] is heading, fc2_out[1] is gate measure
    *heading = tanhf(fc2_out[0]);
    *gate_measure = 1.0f / (1.0f + expf(-fc2_out[1])); // Sigmoid
}