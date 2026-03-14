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
#define CROSS_SIZE       20    
#define CROSS_THICKNESS   3    

/* ------------------------------------------------------------------ */
/* Static CNN input buffer                                             */
/* ------------------------------------------------------------------ */
static float cnn_input[CNN_INPUT_H * CNN_INPUT_W];

/* ------------------------------------------------------------------ */
/* Shared data between camera thread and autopilot thread              */
/* ------------------------------------------------------------------ */
struct gate_cnn_result_t gate_cnn_result = {0.0f, 0.0f, 0, 0};

static volatile float   _shared_heading      = 0.0f;
static volatile float   _shared_gate_measure = 0.0f; // Renamed
static volatile uint8_t _shared_has_gate     = 0;
static volatile uint8_t _new_data_ready      = 0;

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
/* ------------------------------------------------------------------ */
static inline void draw_pixel_yuv422(uint8_t *buf, int img_w, int img_h, int px, int py, uint8_t y_val)
{
    if (px < 0 || px >= img_w || py < 0 || py >= img_h) return;
    buf[py * img_w * 2 + px * 2 + 1] = y_val;
}

/* ------------------------------------------------------------------ */
/* Draw a cross (+) on a YUV422 buffer                                 */
/* ------------------------------------------------------------------ */
static void draw_cross_yuv422(uint8_t *buf, int img_w, int img_h, int cx, int cy, int size, int thick, uint8_t y_val)
{
    int half = thick / 2;
    for (int y = cy - size; y <= cy + size; y++) {
        for (int t = -half; t <= half; t++) {
            draw_pixel_yuv422(buf, img_w, img_h, cx + t, y, y_val);
        }
    }
    for (int x = cx - size; x <= cx + size; x++) {
        for (int t = -half; t <= half; t++) {
            draw_pixel_yuv422(buf, img_w, img_h, x, cy + t, y_val);
        }
    }
}

static void draw_centre_line_yuv422(uint8_t *buf, int img_w, int img_h, uint8_t y_val)
{
    int cy = img_h / 2;
    for (int x = 0; x < img_w; x++) {
        draw_pixel_yuv422(buf, img_w, img_h, x, cy, y_val);
    }
}

/* ------------------------------------------------------------------ */
/* Safe YUV422 extraction with explicit stride parameter               */
/* ------------------------------------------------------------------ */
static void yuv422_to_cnn_input_safe(const uint8_t *src, int stride_bytes, int src_w, int src_h, float *dst)
{
    if (!src || !dst || src_w <= 0 || src_h <= 0) return;

    for (int dy = 0; dy < CNN_INPUT_H; dy++) {
        int sy = (dy * src_h) / CNN_INPUT_H;
        if (sy >= src_h) sy = src_h - 1;
        for (int dx = 0; dx < CNN_INPUT_W; dx++) {
            int sx = (dx * src_w) / CNN_INPUT_W;
            if (sx >= src_w) sx = src_w - 1;
            int y_byte_offset = ((sx / 2) * 4) + (sx % 2 ? 3 : 1);
            uint8_t y_byte = src[sy * stride_bytes + y_byte_offset];
            dst[dy * CNN_INPUT_W + dx] = (float)y_byte * (1.0f / 255.0f);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Camera thread callback                                               */
/* ------------------------------------------------------------------ */
static struct image_t *gate_cnn_func(struct image_t *img, uint8_t camera_id)
{
    (void)camera_id; // Fixes unused parameter warning
    if (img == NULL || img->buf == NULL) return img;
    
    yuv422_to_cnn_input_safe((const uint8_t *)img->buf, img->w * 2, img->w, img->h, cnn_input);
    
    float heading, gate_measure;
    double t0 = now_ms();
    cnn_run(cnn_input, &heading, &gate_measure); // Updated name
    double t1 = now_ms();

    _cnn_tick++;
    _cnn_total_ms += (t1 - t0);
    if (_cnn_tick % CNN_REPORT_EVERY == 0) {
        double avg_ms = _cnn_total_ms / _cnn_tick;
        printf("[gate_cnn] avg inference: %.2f ms (%.1f Hz max)\n", avg_ms, 1000.0 / avg_ms);
    }

    uint8_t has_gate = (gate_measure > gate_cnn_conf_threshold) ? 1 : 0;

    int cross_y = (int)((heading + 1.0f) * 0.5f * (float)img->h);
    int cross_x = img->w / 2;
    if (cross_y < CROSS_SIZE)          cross_y = CROSS_SIZE;
    if (cross_y > img->h - CROSS_SIZE) cross_y = img->h - CROSS_SIZE;

    draw_centre_line_yuv422((uint8_t *)img->buf, img->w, img->h, 100);
    uint8_t cross_y_val = has_gate ? 255 : 80;
    draw_cross_yuv422((uint8_t *)img->buf, img->w, img->h, cross_x, cross_y, CROSS_SIZE, CROSS_THICKNESS, cross_y_val);

    pthread_mutex_lock(&gate_cnn_mutex);
    _shared_heading      = heading;
    _shared_gate_measure = gate_measure;
    _shared_has_gate     = has_gate;
    _new_data_ready      = 1;
    pthread_mutex_unlock(&gate_cnn_mutex);

    return img;
}

/* ------------------------------------------------------------------ */
/* Module init                                                          */
/* ------------------------------------------------------------------ */
void gate_cnn_init(void)
{
    pthread_mutex_init(&gate_cnn_mutex, NULL);
    _new_data_ready      = 0;
    _shared_heading      = 0.0f;
    _shared_gate_measure = 0.0f;
    _shared_has_gate     = 0;

    gate_cnn_result.heading      = 0.0f;
    gate_cnn_result.gate_measure = 0.0f; // Fixed naming and double dots
    gate_cnn_result.has_gate     = 0;
    gate_cnn_result.frame_count  = 0;

    cv_add_to_device(&GATE_CNN_CAMERA, gate_cnn_func, GATE_CNN_FPS, 0);
    printf("gate_cnn: initialised\n");
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

    float   heading      = _shared_heading;
    float   gate_measure = _shared_gate_measure;
    uint8_t has_gate     = _shared_has_gate;
    _new_data_ready      = 0;
    pthread_mutex_unlock(&gate_cnn_mutex);

    gate_cnn_result.heading      = heading;
    gate_cnn_result.gate_measure = gate_measure; // Fixed naming
    gate_cnn_result.has_gate     = has_gate;
    gate_cnn_result.frame_count++;

    int16_t pixel_x = (int16_t)((heading + 1.0f) * 500.0f);
    int16_t pixel_y = 500;
    int16_t size_w  = 0;
    int16_t size_h  = 0;
    int32_t quality = (int32_t)(gate_measure * 255.0f);
    int16_t extra   = has_gate;

    AbiSendMsgVISUAL_DETECTION(GATE_CNN_ABI_ID, pixel_x, pixel_y, size_w, size_h, quality, extra);
}