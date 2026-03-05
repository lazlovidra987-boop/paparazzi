// Incluimos nuestro propio header primero
#include "modules/computer_vision/obj_detection_std.h"

// Headers del sistema y framework de visión
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "std.h"

#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "pthread.h"

// Definimos un ID para nuestro módulo. 
// ABI_BROADCAST significa "enviar a cualquiera que esté escuchando".
#ifndef EDGE_DETECTION_ID
#define EDGE_DETECTION_ID ABI_BROADCAST 
#endif

// 1. DEFINICIÓN DE VARIABLES GLOBALES (Declaradas como extern en el .h)
uint16_t edge_block_size = 10;
float edge_std_dev_threshold = 40.0f;
bool edge_draw_edges = true;

// 2. VARIABLES PRIVADAS DEL MÓDULO
static pthread_mutex_t mutex;

struct edge_detection_t {
    uint32_t blocks_detected;
    bool updated;
};
static struct edge_detection_t global_edge_state;

// 3. FUNCIONES AUXILIARES
/*
 * Extrae el puntero directo a la luminancia (canal Y) de un píxel en formato YUV422
 */
static inline uint8_t* get_y_ptr(struct image_t *img, int x, int y) {
    return &((uint8_t*)img->buf)[y * 2 * img->w + 2 * x + 1];
}

/*
 * Núcleo matemático: Recorre por bloques y calcula la desviación estándar
 */
static uint32_t detect_edges_stddev(struct image_t *img, uint16_t block_size, float threshold, bool draw) {
    uint32_t detected_blocks = 0;

    for (int y = 0; y < img->h; y += block_size) {
        for (int x = 0; x < img->w; x += block_size) {
            
            int w = (x + block_size > img->w) ? (img->w - x) : block_size;
            int h = (y + block_size > img->h) ? (img->h - y) : block_size;
            int num_pixels = w * h;

            if (num_pixels == 0) continue;

            // Calcular media
            float sum = 0;
            for (int by = 0; by < h; by++) {
                for (int bx = 0; bx < w; bx++) {
                    sum += *get_y_ptr(img, x + bx, y + by);
                }
            }
            float mean = sum / num_pixels;

            // Calcular varianza
            float sum_diff_sq = 0;
            for (int by = 0; by < h; by++) {
                for (int bx = 0; bx < w; bx++) {
                    float val = *get_y_ptr(img, x + bx, y + by);
                    sum_diff_sq += (val - mean) * (val - mean);
                }
            }
            
            float std_dev = sqrtf(sum_diff_sq / num_pixels);

            // Umbralización y dibujo
            if (std_dev > threshold) {
                detected_blocks++;

                if (draw) {
                    // Dibujar rectángulo (saturando el brillo a 255 - blanco puro)
                    for (int bx = 0; bx < w; bx++) {
                        *get_y_ptr(img, x + bx, y) = 255; // Arriba
                        if (h > 1) *get_y_ptr(img, x + bx, y + h - 1) = 255; // Abajo
                    }
                    for (int by = 0; by < h; by++) {
                        *get_y_ptr(img, x, y + by) = 255; // Izquierda
                        if (w > 1) *get_y_ptr(img, x + w - 1, y + by) = 255; // Derecha
                    }
                }
            }
        }
    }
    return detected_blocks;
}

// 4. FUNCIONES DEL PIPELINE DE VIDEO (CALLBACKS)
/*
 * Función principal que procesa la imagen entrante
 */
static struct image_t *edge_detector_cb(struct image_t *img) {
    // Leemos las variables globales que podrían haber sido cambiadas por el usuario
    uint32_t blocks = detect_edges_stddev(img, edge_block_size, edge_std_dev_threshold, edge_draw_edges);

    // Guardado seguro usando el Mutex
    pthread_mutex_lock(&mutex);
    global_edge_state.blocks_detected = blocks;
    global_edge_state.updated = true;
    pthread_mutex_unlock(&mutex);

    return img;
}

/* * Envoltorio para cumplir con el tipo cv_function (struct image_t * (*)(struct image_t *, uint8_t))
 */
static struct image_t *edge_detector_wrapper(struct image_t *img, uint8_t camera_id __attribute__((unused))) {
    return edge_detector_cb(img);
}

// 5. FUNCIONES DE CICLO DE VIDA DEL SISTEMA (INIT Y PERIODIC)
/*
 * Se llama una vez al encender el sistema
 */
void edge_detector_init(void) {
    // Inicializar estado seguro
    memset(&global_edge_state, 0, sizeof(struct edge_detection_t));
    pthread_mutex_init(&mutex, NULL);

    // Suscribir nuestra función a la cámara (Asumiendo que CAMERA_DEVICE está definida en tu config)
    // El '0' en FPS significa que correrá a la máxima velocidad que de la cámara.
#ifdef EDGE_DETECTOR_CAMERA
    cv_add_to_device(&EDGE_DETECTOR_CAMERA, edge_detector_wrapper, 0, 0);
#endif
//#ifdef FRONT_CAMERA
  //  cv_add_to_device(&FRONT_CAMERA, edge_detector_wrapper, 0, 0);
//#endif
}

/*
 * Se llama a una frecuencia fija (ej. 10Hz) por el piloto automático
 */
void edge_detector_periodic(void) {
    struct edge_detection_t local_state;
    local_state.updated = false;

    // Entramos a la zona crítica rápido para copiar y salir
    pthread_mutex_lock(&mutex);
    if (global_edge_state.updated) {
        local_state.blocks_detected = global_edge_state.blocks_detected;
        local_state.updated = true;
        
        // Reseteamos la bandera global para no reenviar datos viejos
        global_edge_state.updated = false; 
    }
    pthread_mutex_unlock(&mutex);

    // Si hubo una foto nueva procesada, enviamos el mensaje al sistema
    if (local_state.updated) {
        AbiSendMsgVISUAL_DETECTION(EDGE_DETECTION_ID, 0, 0, 0, 0, local_state.blocks_detected, 0);
    }
}