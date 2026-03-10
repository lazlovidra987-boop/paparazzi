#include "cnn.h"
#include "model_weights.h"

#include <math.h>
#include <stddef.h>

/* ============================================================
 * CNN MODEL SHAPES
 * Input: [1, 1, 520, 240]
 *
 * Layer shapes:
 *   conv1   -> [8, 520, 240]
 *   pool1   -> [8, 260, 240]   kernel=(2,1)
 *   conv2   -> [16, 260, 240]
 *   pool2   -> [16, 130, 120]  kernel=(2,2)
 *   conv3   -> [32, 130, 120]
 *   pool3   -> [32,  65,  60]  kernel=(2,2)
 *   conv4   -> [32,  65,  60]
 *   avgpool -> [32]
 *   fc1     -> [16]
 *   fc2     -> [1]
 * ============================================================ */

#define IN_H   520
#define IN_W   240

#define C1_OC  8
#define C2_OC  16
#define C3_OC  32
#define C4_OC  32

#define P1_H   260
#define P1_W   240

#define P2_H   130
#define P2_W   120

#define P3_H   65
#define P3_W   60

#define FC1_OUT 16
#define FC2_OUT 1

#define BUF_A_SIZE 998400
#define BUF_B_SIZE 499200

static float buf_a[BUF_A_SIZE];
static float buf_b[BUF_B_SIZE];
static float gap_out[32];
static float fc1_out[FC1_OUT];
static float fc2_out[FC2_OUT];

static inline float sigmoidf_fast(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static void relu_inplace(float *x, int n)
{
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0f) {
            x[i] = 0.0f;
        }
    }
}

static void conv2d_same_3x3(
    const float *input,
    int in_c,
    int in_h,
    int in_w,
    const float *weight,
    const float *bias,
    int out_c,
    float *output)
{
    for (int oc = 0; oc < out_c; oc++) {
        for (int y = 0; y < in_h; y++) {
            for (int x = 0; x < in_w; x++) {

                float sum = bias[oc];

                for (int ic = 0; ic < in_c; ic++) {
                    for (int ky = 0; ky < 3; ky++) {
                        int iy = y + ky - 1;
                        if (iy < 0 || iy >= in_h) {
                            continue;
                        }

                        for (int kx = 0; kx < 3; kx++) {
                            int ix = x + kx - 1;
                            if (ix < 0 || ix >= in_w) {
                                continue;
                            }

                            int in_idx =
                                (ic * in_h + iy) * in_w + ix;

                            int w_idx =
                                (((oc * in_c + ic) * 3 + ky) * 3 + kx);

                            sum += input[in_idx] * weight[w_idx];
                        }
                    }
                }

                int out_idx =
                    (oc * in_h + y) * in_w + x;

                output[out_idx] = sum;
            }
        }
    }
}

static void maxpool2d_2x1(
    const float *input,
    int c,
    int in_h,
    int in_w,
    float *output)
{
    int out_h = in_h / 2;
    int out_w = in_w;

    for (int ch = 0; ch < c; ch++) {
        for (int y = 0; y < out_h; y++) {
            int iy0 = 2 * y;
            int iy1 = iy0 + 1;

            for (int x = 0; x < out_w; x++) {
                int idx0 = (ch * in_h + iy0) * in_w + x;
                int idx1 = (ch * in_h + iy1) * in_w + x;

                float a = input[idx0];
                float b = input[idx1];
                output[(ch * out_h + y) * out_w + x] = (a > b) ? a : b;
            }
        }
    }
}

static void maxpool2d_2x2(
    const float *input,
    int c,
    int in_h,
    int in_w,
    float *output)
{
    int out_h = in_h / 2;
    int out_w = in_w / 2;

    for (int ch = 0; ch < c; ch++) {
        for (int y = 0; y < out_h; y++) {
            int iy0 = 2 * y;
            int iy1 = iy0 + 1;

            for (int x = 0; x < out_w; x++) {
                int ix0 = 2 * x;
                int ix1 = ix0 + 1;

                float v0 = input[(ch * in_h + iy0) * in_w + ix0];
                float v1 = input[(ch * in_h + iy0) * in_w + ix1];
                float v2 = input[(ch * in_h + iy1) * in_w + ix0];
                float v3 = input[(ch * in_h + iy1) * in_w + ix1];

                float m01 = (v0 > v1) ? v0 : v1;
                float m23 = (v2 > v3) ? v2 : v3;
                float m   = (m01 > m23) ? m01 : m23;

                output[(ch * out_h + y) * out_w + x] = m;
            }
        }
    }
}

static void global_avgpool(
    const float *input,
    int c,
    int h,
    int w,
    float *output)
{
    const int hw = h * w;
    const float inv_hw = 1.0f / (float)hw;

    for (int ch = 0; ch < c; ch++) {
        float sum = 0.0f;
        const float *base = input + ch * hw;

        for (int i = 0; i < hw; i++) {
            sum += base[i];
        }

        output[ch] = sum * inv_hw;
    }
}

static void linear(
    const float *input,
    int in_features,
    const float *weight,
    const float *bias,
    int out_features,
    float *output)
{
    for (int o = 0; o < out_features; o++) {
        float sum = bias[o];

        for (int i = 0; i < in_features; i++) {
            sum += weight[o * in_features + i] * input[i];
        }

        output[o] = sum;
    }
}

void cnn_forward(const float *input, float *output)
{
    conv2d_same_3x3(
        input,
        1, IN_H, IN_W,
        features_0_weight,
        features_0_bias,
        C1_OC,
        buf_a
    );
    relu_inplace(buf_a, C1_OC * IN_H * IN_W);

    maxpool2d_2x1(
        buf_a,
        C1_OC, IN_H, IN_W,
        buf_b
    );

    conv2d_same_3x3(
        buf_b,
        8, P1_H, P1_W,
        features_3_weight,
        features_3_bias,
        C2_OC,
        buf_a
    );
    relu_inplace(buf_a, C2_OC * P1_H * P1_W);

    maxpool2d_2x2(
        buf_a,
        C2_OC, P1_H, P1_W,
        buf_b
    );

    conv2d_same_3x3(
        buf_b,
        16, P2_H, P2_W,
        features_6_weight,
        features_6_bias,
        C3_OC,
        buf_a
    );
    relu_inplace(buf_a, C3_OC * P2_H * P2_W);

    maxpool2d_2x2(
        buf_a,
        C3_OC, P2_H, P2_W,
        buf_b
    );

    conv2d_same_3x3(
        buf_b,
        32, P3_H, P3_W,
        features_9_weight,
        features_9_bias,
        C4_OC,
        buf_a
    );
    relu_inplace(buf_a, C4_OC * P3_H * P3_W);

    global_avgpool(
        buf_a,
        32, P3_H, P3_W,
        gap_out
    );

    linear(
        gap_out,
        32,
        head_1_weight,
        head_1_bias,
        16,
        fc1_out
    );
    relu_inplace(fc1_out, 16);

    linear(
        fc1_out,
        16,
        head_3_weight,
        head_3_bias,
        1,
        fc2_out
    );

    output[0] = sigmoidf_fast(fc2_out[0]);
}