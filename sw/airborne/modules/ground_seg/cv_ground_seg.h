#ifndef CV_GROUND_SEG_H
#define CV_GROUND_SEG_H

#include "modules/computer_vision/cv.h"
#include "pthread.h"

// Define module-specific constants and configuration
#define GROUND_SEGMENTATION_FPS_DEFAULT 0 // 0 means run at camera fps
#define GROUND_SEGMENTATION_CAMERA_DEFAULT "front_camera"

// Structure to hold the ground detection results
struct ground_detection_t {
    int32_t x_c;         // x coordinate of the centroid of the detected ground
    int32_t y_c;         // y coordinate of the centroid of the detected ground
    uint32_t color_count; // Number of ground pixels detected
    bool updated;         // Flag to indicate if new data is available
};

// Filter Settings for Ground (Green Color Range in YUV)
extern uint8_t ground_lum_min;    // Y (brightness) minimum threshold
extern uint8_t ground_lum_max;   // Y (brightness) maximum threshold
extern uint8_t ground_cb_min;     // U (chrominance) minimum threshold
extern uint8_t ground_cb_max;    // U (chrominance) maximum threshold
extern uint8_t ground_cr_min;     // V (chrominance) minimum threshold
extern uint8_t ground_cr_max;    // V (chrominance) maximum threshold

// Declare the global ground detection structure
extern struct ground_detection_t global_ground_data;

// Function declarations for the ground segmentation module
extern void ground_segmentation_init(void); // Initialization function
// extern struct image_t *ground_segmentation(struct image_t *img); // Main processing function
extern void ground_segmentation_periodic(void); // Periodic function for data handling

// Module functions
// extern void color_object_detector_init(void);
// extern void color_object_detector_periodic(void);

#endif /* CV_GROUND_SEG_H */