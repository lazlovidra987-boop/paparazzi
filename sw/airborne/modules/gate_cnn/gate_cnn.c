/*
 * gate_cnn.c
 * ----------
 * Paparazzi module: CNN-based gate detector for the Bebop drone.
 *
 * How it works
 * ------------
 * 1. init:     Register gate_cnn_func as a video callback on GATE_CNN_CAMERA.
 *              Initialise the mutex that protects shared data.
 *
 * 2. callback  (runs in the camera thread, NOT the autopilot thread):
 *   a. Extract the Y (luminance) channel from the YUV422 image.
 *   b. Resize from camera resolution to CNN input [120 x 160].
 *   c. Normalise pixels from [0,255] to [0.0, 1.0].
 *   d. Call cnn_run() -> heading, confidence.
 *   e. Lock mutex, copy results to shared struct, set flag, unlock.
 *
 * 3. periodic  (runs in the autopilot thread at GATE_CNN_FPS Hz):
 *   a. Lock mutex, check flag, copy results, clear flag, unlock.
 *   b. Publish ABI VISUAL_DETECTION message.
 *
 * YUV422 (UYVY) layout
 * --------------------
 * Bytes:  U0 Y0 V0 Y1  U2 Y2 V2 Y3  ...
 * Y values are at byte indices 1, 3, 5, 7, ... (every 2nd byte, offset 1).
 * So for pixel (row, col):
 *   byte_index = row * stride + col * 2 + 1
 * where stride = image_width * 2.
 *
 * Resize (nearest-neighbour)
 * --------------------------
 * We downsample from camera resolution (typically 520x240 or 640x480)
 * to CNN_INPUT_H x CNN_INPUT_W (120 x 160) using nearest-neighbour
 * sampling — fast and good enough for a CNN input.
 */

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

/* Paparazzi includes */
#include "modules/core/abi.h"
#include "modules/computer_vision/cv.h"

/* Our module */
#include "gate_cnn.h"
#include "cnn_inference.h"   /* cnn_run(), CNN_INPUT_H, CNN_INPUT_W */


/* ------------------------------------------------------------------ */
/* Static CNN input buffer (normalised floats, [0, 1])                 */
/* ------------------------------------------------------------------ */
static float cnn_input[CNN_INPUT_H * CNN_INPUT_W];


/* ------------------------------------------------------------------ */
/* Shared data between camera thread and autopilot thread              */
/* ------------------------------------------------------------------ */
struct gate_cnn_result_t gate_cnn_result = {0.0f, 0.0f, 0, 0};

static volatile float   _shared_heading    = 0.0f;
static volatile float   _shared_confidence = 0.0f;
static volatile uint8_t _shared_has_gate   = 0;
static volatile uint8_t _new_data_ready    = 0;

static pthread_mutex_t gate_cnn_mutex;
float gate_cnn_conf_threshold = GATE_CNN_CONF_THRESHOLD; 


/* Timing */
static double _cnn_total_ms  = 0.0;
static uint32_t _cnn_tick    = 0;
#define CNN_REPORT_EVERY 100

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* ------------------------------------------------------------------ */
/* Nearest-neighbour resize + Y-channel extract                        */
/*                                                                     */
/* src      : raw YUV422 (UYVY) image buffer                           */
/* src_w    : source image width  in pixels                            */
/* src_h    : source image height in pixels                            */
/* dst      : output float buffer [CNN_INPUT_H * CNN_INPUT_W]          */
/* ------------------------------------------------------------------ */
static void yuv422_to_cnn_input(
    const uint8_t *src,
    int            src_w,
    int            src_h,
    float         *dst)
{
    for (int dy = 0; dy < CNN_INPUT_H; dy++) {
        /* Map output row -> source row (nearest neighbour) */
        int sy = (dy * src_h) / CNN_INPUT_H;

        for (int dx = 0; dx < CNN_INPUT_W; dx++) {
            /* Map output col -> source col (nearest neighbour) */
            int sx = (dx * src_w) / CNN_INPUT_W;

            /* In UYVY, Y byte for pixel (sy, sx):
             * byte index = sy * src_w * 2 + sx * 2 + 1  */
            uint8_t y_byte = src[sy * src_w * 2 + sx * 2 + 1];

            dst[dy * CNN_INPUT_W + dx] = (float)y_byte / 255.0f;
        }
    }
}


/* ------------------------------------------------------------------ */
/* Camera thread callback                                               */
/* Called by the video driver for every new frame.                     */
/* Must return quickly — heavy work is OK here because it runs in its  */
/* own thread, not the autopilot thread.                                */
/* ------------------------------------------------------------------ */
static struct image_t *gate_cnn_func(struct image_t *img,
                                     uint8_t camera_id __attribute__((unused)))
{
    if (img == NULL || img->buf == NULL) {
        return img;
    }

    /* 1. Extract Y channel and resize to CNN input size */
    yuv422_to_cnn_input(
        (const uint8_t *)img->buf,
        img->w,
        img->h,
        cnn_input
    );

    /* 2. Run CNN forward pass */
    /* 2. Run CNN forward pass */
    float heading, confidence;
    double t0 = now_ms();
    cnn_run(cnn_input, &heading, &confidence);
    double t1 = now_ms();

    _cnn_tick++;
    _cnn_total_ms += (t1 - t0);
    if (_cnn_tick % CNN_REPORT_EVERY == 0) {
        double avg_ms = _cnn_total_ms / _cnn_tick;
        printf("[gate_cnn] avg inference: %.2f ms  (%.1f Hz max)  frames=%u\n",
               avg_ms, 1000.0 / avg_ms, _cnn_tick);
    }
    /* 3. Threshold confidence */
    uint8_t has_gate = (confidence > gate_cnn_conf_threshold) ? 1 : 0;

    /* 4. Copy to shared variables (mutex-protected) */
    pthread_mutex_lock(&gate_cnn_mutex);
    _shared_heading    = heading;
    _shared_confidence = confidence;
    _shared_has_gate   = has_gate;
    _new_data_ready    = 1;
    pthread_mutex_unlock(&gate_cnn_mutex);

    return img;
}


/* ------------------------------------------------------------------ */
/* Module init — called once at startup                                 */
/* ------------------------------------------------------------------ */
void gate_cnn_init(void)
{
    pthread_mutex_init(&gate_cnn_mutex, NULL);

    _new_data_ready    = 0;
    _shared_heading    = 0.0f;
    _shared_confidence = 0.0f;
    _shared_has_gate   = 0;

    gate_cnn_result.heading     = 0.0f;
    gate_cnn_result.confidence  = 0.0f;
    gate_cnn_result.has_gate    = 0;
    gate_cnn_result.frame_count = 0;

    /* Register video callback */
    cv_add_to_device(&GATE_CNN_CAMERA, gate_cnn_func, GATE_CNN_FPS, 0);

    printf("gate_cnn: initialised (ABI id=%d, threshold=%.2f, fps=%d)\n",
           GATE_CNN_ABI_ID, GATE_CNN_CONF_THRESHOLD, GATE_CNN_FPS);
}


/* ------------------------------------------------------------------ */
/* Module periodic — called at GATE_CNN_FPS Hz by the autopilot        */
/* ------------------------------------------------------------------ */
void gate_cnn_periodic(void)
{
    /* Read shared data if a new frame has been processed */
    pthread_mutex_lock(&gate_cnn_mutex);
    if (!_new_data_ready) {
        pthread_mutex_unlock(&gate_cnn_mutex);
        return;
    }

    float   heading    = _shared_heading;
    float   confidence = _shared_confidence;
    uint8_t has_gate   = _shared_has_gate;
    _new_data_ready    = 0;
    pthread_mutex_unlock(&gate_cnn_mutex);

    /* Update public result struct */
    gate_cnn_result.heading     = heading;
    gate_cnn_result.confidence  = confidence;
    gate_cnn_result.has_gate    = has_gate;
    gate_cnn_result.frame_count++;

    /*
     * Publish ABI VISUAL_DETECTION message.
     *
     * Field mapping:
     *   pixel_x   = heading mapped to pixel coordinates [0, 1000]
     *               (centre = 500, left = 0, right = 1000)
     *   pixel_y   = 500 (unknown vertical position)
     *   pixel_w   = 0   (unknown width)
     *   pixel_h   = 0   (unknown height)
     *   quality   = confidence scaled to int [0, 255]
     *   extra     = 0
     *
     * Receivers can decode heading back with:
     *   heading = (pixel_x - 500) / 500.0f
     */
    int16_t pixel_x = (int16_t)((heading + 1.0f) * 500.0f);   /* [0, 1000] */
    int16_t pixel_y = 500;
    int16_t size_w  = 0;
    int16_t size_h  = 0;
    int32_t quality = (int32_t)(confidence * 255.0f);
    int16_t extra   = has_gate;

    AbiSendMsgVISUAL_DETECTION(
        GATE_CNN_ABI_ID,
        pixel_x,
        pixel_y,
        size_w,
        size_h,
        quality,
        extra
    );
}