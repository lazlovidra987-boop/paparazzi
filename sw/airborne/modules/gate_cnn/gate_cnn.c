#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>

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
/* Threading and Buffering                                             */
/* ------------------------------------------------------------------ */
static float cnn_input[CNN_INPUT_H * CNN_INPUT_W];
static pthread_t cnn_thread;
static pthread_mutex_t cnn_mutex;
static pthread_cond_t cnn_cond;
static volatile bool cnn_is_busy = false;
static volatile bool cnn_new_data_available = false;

/* Shared data for the Autopilot thread */
struct gate_cnn_result_t gate_cnn_result = {0.0f, 0.0f, 0, 0};
static volatile float   _shared_heading      = 0.0f;
static volatile float   _shared_gate_measure = 0.0f;
static volatile uint8_t _shared_has_gate     = 0;
static volatile uint8_t _new_autopilot_data  = 0;

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
/* Drawing Utilities (YUV422)                                         */
/* ------------------------------------------------------------------ */
static inline void draw_pixel_yuv422(uint8_t *buf, int img_w, int img_h, int px, int py, uint8_t y_val)
{
    if (px < 0 || px >= img_w || py < 0 || py >= img_h) return;
    buf[py * img_w * 2 + px * 2 + 1] = y_val;
}

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

/* ------------------------------------------------------------------ */
/* Extraction                                                         */
/* ------------------------------------------------------------------ */
static void yuv422_to_cnn_input_safe(const uint8_t *src, int stride_bytes, int src_w, int src_h, float *dst)
{
    if (!src || !dst || src_w <= 0 || src_h <= 0) return;
    for (int dy = 0; dy < CNN_INPUT_H; dy++) {
        int sy = (dy * src_h) / CNN_INPUT_H;
        for (int dx = 0; dx < CNN_INPUT_W; dx++) {
            int sx = (dx * src_w) / CNN_INPUT_W;
            int y_byte_offset = ((sx / 2) * 4) + (sx % 2 ? 3 : 1);
            uint8_t y_byte = src[sy * stride_bytes + y_byte_offset];
            dst[dy * CNN_INPUT_W + dx] = (float)y_byte * (1.0f / 255.0f);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Background Inference Thread                                         */
/* ------------------------------------------------------------------ */
static void *gate_cnn_thread_main(void *data)
{
    (void)data;
    float h, m;
    while (1) {
        pthread_mutex_lock(&cnn_mutex);
        while (!cnn_new_data_available) {
            pthread_cond_wait(&cnn_cond, &cnn_mutex);
        }
        cnn_new_data_available = false;
        pthread_mutex_unlock(&cnn_mutex);

        // Actual heavy math
        double t0 = now_ms();
        cnn_run(cnn_input, &h, &m);
        double t1 = now_ms();

        _cnn_tick++;
        _cnn_total_ms += (t1 - t0);
        if (_cnn_tick % CNN_REPORT_EVERY == 0) {
            printf("[gate_cnn] avg: %.2f ms\n", _cnn_total_ms / _cnn_tick);
        }

        // Push results to shared vars for autopilot
        pthread_mutex_lock(&cnn_mutex);
        _shared_heading = h;
        _shared_gate_measure = m;
        _shared_has_gate = (m > gate_cnn_conf_threshold) ? 1 : 0;
        _new_autopilot_data = 1;
        cnn_is_busy = false; // Reset busy flag ONLY when done
        pthread_mutex_unlock(&cnn_mutex);
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Camera Thread Callback (called by video_thread)                    */
/* ------------------------------------------------------------------ */
static struct image_t *gate_cnn_func(struct image_t *img, uint8_t camera_id)
{
    (void)camera_id;
    if (img == NULL || img->buf == NULL) return img;

    /* 1. Check if we are busy. If so, return IMMEDIATELY. 
          This prevents the Select Timeout by not blocking the video thread. */
    if (cnn_is_busy) {
        return img;
    }

    /* 2. Try to get the lock. Use trylock so we NEVER wait here. */
    if (pthread_mutex_trylock(&gate_cnn_mutex) == 0) {
        cnn_is_busy = true; // Mark as busy BEFORE starting the heavy work

        // Copy image to CNN buffer
        yuv422_to_cnn_input_safe((const uint8_t *)img->buf, img->w * 2, img->w, img->h, cnn_input);
        
        float heading, gate_measure;
        // The heavy math happens here:
        cnn_run(cnn_input, &heading, &gate_measure); 

        // Update shared variables
        _shared_heading = heading;
        _shared_gate_measure = gate_measure;
        _shared_has_gate = (gate_measure > gate_cnn_conf_threshold) ? 1 : 0;
        _new_data_ready = 1;

        cnn_is_busy = false; // Mark as NOT busy when finished
        pthread_mutex_unlock(&gate_cnn_mutex);
    }

    return img;
}

void gate_cnn_init(void)
{
    pthread_mutex_init(&cnn_mutex, NULL);
    pthread_cond_init(&cnn_cond, NULL);
    
    pthread_create(&cnn_thread, NULL, gate_cnn_thread_main, NULL);

    cv_add_to_device(&GATE_CNN_CAMERA, gate_cnn_func, GATE_CNN_FPS, 0);
    printf("gate_cnn: async thread initialised\n");
}

void gate_cnn_periodic(void)
{
    if (pthread_mutex_trylock(&cnn_mutex) == 0) {
        if (_new_autopilot_data) {
            gate_cnn_result.heading = _shared_heading;
            gate_cnn_result.gate_measure = _shared_gate_measure;
            gate_cnn_result.has_gate = _shared_has_gate;
            gate_cnn_result.frame_count++;
            _new_autopilot_data = 0;

            int16_t px = (int16_t)((gate_cnn_result.heading + 1.0f) * 500.0f);
            AbiSendMsgVISUAL_DETECTION(GATE_CNN_ABI_ID, px, 500, 0, 0, (int32_t)(gate_cnn_result.gate_measure * 255.0f), gate_cnn_result.has_gate);
        }
        pthread_mutex_unlock(&cnn_mutex);
    }
}