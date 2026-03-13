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
 *   e. Draw cross on the live frame at predicted gate position.
 *   f. Lock mutex, copy results to shared struct, set flag, unlock.
 *
 * 3. periodic  (runs in the autopilot thread at GATE_CNN_FPS Hz):
 *   a. Lock mutex, check flag, copy results, clear flag, unlock.
 *   b. Publish ABI VISUAL_DETECTION message.
 *
 * Cross overlay
 * -------------
 * The cross is drawn directly on the YUV422 buffer so it appears in
 * the RTP video stream viewed in Paparazzi GCS.
 *
 * The camera is physically rotated 90 degrees, so heading maps to the
 * VERTICAL axis of the image:
 *   cross_y = (heading + 1.0) / 2.0 * img->h
 *   cross_x = img->w / 2
 *
 * Colours in YUV422 (UYVY):
 *   White  : Y=255, U=128, V=128
 *   Green  : Y=150, U=44,  V=21
 *   Red    : Y=76,  U=84,  V=255
 *
 * YUV422 (UYVY) layout
 * --------------------
 * Bytes:  U0 Y0 V0 Y1  U2 Y2 V2 Y3  ...
 * Y values are at byte indices 1, 3, 5, 7, ... (every 2nd byte, offset 1).
 * So for pixel (row, col):
 *   byte_index = row * stride + col * 2 + 1
 * where stride = image_width * 2.
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
/* Cross drawing size                                                   */
/* ------------------------------------------------------------------ */
#define CROSS_SIZE       20    /* half-length of each arm in pixels   */
#define CROSS_THICKNESS   3    /* line thickness in pixels             */


/* ------------------------------------------------------------------ */
/* Static CNN input buffer                                             */
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
static double   _cnn_total_ms = 0.0;
static uint32_t _cnn_tick     = 0;
#define CNN_REPORT_EVERY 100

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}


/* ------------------------------------------------------------------ */
/* Draw one pixel on a YUV422 (UYVY) buffer                            */
/*                                                                     */
/* Sets Y value only — fast and sufficient for a visible cross.        */
/* y_val: 255 = white, 0 = black, 76 = dark (use with chroma for red) */
/* ------------------------------------------------------------------ */
static inline void draw_pixel_yuv422(
    uint8_t *buf,
    int      img_w,
    int      img_h,
    int      px,
    int      py,
    uint8_t  y_val)
{
    if (px < 0 || px >= img_w || py < 0 || py >= img_h) return;
    /* Y byte for pixel (py, px) is at: py * img_w * 2 + px * 2 + 1 */
    buf[py * img_w * 2 + px * 2 + 1] = y_val;
}


/* ------------------------------------------------------------------ */
/* Draw a cross (+) on a YUV422 buffer                                 */
/*                                                                     */
/* cx, cy : centre of cross in image pixels                            */
/* size   : half-length of each arm                                    */
/* thick  : line thickness                                             */
/* y_val  : brightness (255=white, 0=black)                            */
/* ------------------------------------------------------------------ */
static void draw_cross_yuv422(
    uint8_t *buf,
    int      img_w,
    int      img_h,
    int      cx,
    int      cy,
    int      size,
    int      thick,
    uint8_t  y_val)
{
    int half = thick / 2;

    /* Vertical arm */
    for (int y = cy - size; y <= cy + size; y++) {
        for (int t = -half; t <= half; t++) {
            draw_pixel_yuv422(buf, img_w, img_h, cx + t, y, y_val);
        }
    }

    /* Horizontal arm */
    for (int x = cx - size; x <= cx + size; x++) {
        for (int t = -half; t <= half; t++) {
            draw_pixel_yuv422(buf, img_w, img_h, x, cy + t, y_val);
        }
    }
}


/* ------------------------------------------------------------------ */
/* Draw a horizontal centre line (heading = 0 reference)               */
/* ------------------------------------------------------------------ */
static void draw_centre_line_yuv422(
    uint8_t *buf,
    int      img_w,
    int      img_h,
    uint8_t  y_val)
{
    int cy = img_h / 2;
    for (int x = 0; x < img_w; x++) {
        draw_pixel_yuv422(buf, img_w, img_h, x, cy, y_val);
    }
}


/* ------------------------------------------------------------------ */
/* Nearest-neighbour resize + Y-channel extract                        */
/* ------------------------------------------------------------------ */
static void yuv422_to_cnn_input(
    const uint8_t *src,
    int            src_w,
    int            src_h,
    float         *dst)
{
    for (int dy = 0; dy < CNN_INPUT_H; dy++) {
        int sy = (dy * src_h) / CNN_INPUT_H;
        for (int dx = 0; dx < CNN_INPUT_W; dx++) {
            int     sx     = (dx * src_w) / CNN_INPUT_W;
            uint8_t y_byte = src[sy * src_w * 2 + sx * 2 + 1];
            dst[dy * CNN_INPUT_W + dx] = (float)y_byte / 255.0f;
        }
    }
}


/* ------------------------------------------------------------------ */
/* Camera thread callback                                               */
/* ------------------------------------------------------------------ */
static struct image_t *gate_cnn_func(struct image_t *img,
                                     uint8_t camera_id __attribute__((unused)))
{
    if (img == NULL || img->buf == NULL) {
        return img;
    }

    /* 1. Extract Y channel and resize to CNN input */
    yuv422_to_cnn_input(
        (const uint8_t *)img->buf,
        img->w,
        img->h,
        cnn_input
    );

    /* 2. Run CNN forward pass */
    float  heading, confidence;
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

    /* 3. Threshold */
    uint8_t has_gate = (confidence > gate_cnn_conf_threshold) ? 1 : 0;

    /* 4. Draw cross on the live frame
     *
     * Camera is rotated 90 deg: heading is on the VERTICAL axis.
     *   heading = -1  ->  top    of image  ->  drone's left
     *   heading =  0  ->  centre of image  ->  straight ahead
     *   heading = +1  ->  bottom of image  ->  drone's right
     *
     * cross_y = (heading + 1) / 2 * img->h
     * cross_x = img->w / 2  (horizontal position unknown)
     */
    int cross_y = (int)((heading + 1.0f) * 0.5f * (float)img->h);
    int cross_x = img->w / 2;

    /* Clamp to image bounds */
    if (cross_y < CROSS_SIZE)          cross_y = CROSS_SIZE;
    if (cross_y > img->h - CROSS_SIZE) cross_y = img->h - CROSS_SIZE;

    /* Draw faint centre reference line (Y=100 = dark grey) */
    draw_centre_line_yuv422(
        (uint8_t *)img->buf, img->w, img->h, 100);

    /* Draw cross: white (Y=255) if gate detected, dark (Y=80) if not */
    uint8_t cross_y_val = has_gate ? 255 : 80;
    draw_cross_yuv422(
        (uint8_t *)img->buf,
        img->w, img->h,
        cross_x, cross_y,
        CROSS_SIZE, CROSS_THICKNESS,
        cross_y_val
    );

    /* 5. Copy to shared variables */
    pthread_mutex_lock(&gate_cnn_mutex);
    _shared_heading    = heading;
    _shared_confidence = confidence;
    _shared_has_gate   = has_gate;
    _new_data_ready    = 1;
    pthread_mutex_unlock(&gate_cnn_mutex);

    return img;
}


/* ------------------------------------------------------------------ */
/* Module init                                                          */
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

    cv_add_to_device(&GATE_CNN_CAMERA, gate_cnn_func, GATE_CNN_FPS, 0);

    printf("gate_cnn: initialised (ABI id=%d, threshold=%.2f, fps=%d)\n",
           GATE_CNN_ABI_ID, GATE_CNN_CONF_THRESHOLD, GATE_CNN_FPS);
}


/* ------------------------------------------------------------------ */
/* Module periodic                                                      */
/* ------------------------------------------------------------------ */
void gate_cnn_periodic(void)
{
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

    gate_cnn_result.heading     = heading;
    gate_cnn_result.confidence  = confidence;
    gate_cnn_result.has_gate    = has_gate;
    gate_cnn_result.frame_count++;

    int16_t pixel_x = (int16_t)((heading + 1.0f) * 500.0f);
    int16_t pixel_y = 500;
    int16_t size_w  = 0;
    int16_t size_h  = 0;
    int32_t quality = (int32_t)(confidence * 255.0f);
    int16_t extra   = has_gate;

    AbiSendMsgVISUAL_DETECTION(
        GATE_CNN_ABI_ID,
        pixel_x, pixel_y,
        size_w,  size_h,
        quality, extra
    );
}