#ifndef EDGE_DETECTOR_CV_H
#define EDGE_DETECTOR_CV_H

#include <stdint.h>
#include <stdbool.h>

// Configuraciones del módulo (extern permite que se modifiquen desde fuera)
extern uint16_t edge_block_size;
extern float edge_std_dev_threshold;
extern bool edge_draw_edges;

// Funciones principales del módulo
extern void edge_detector_init(void);
extern void edge_detector_periodic(void);

#endif /* EDGE_DETECTOR_CV_H */