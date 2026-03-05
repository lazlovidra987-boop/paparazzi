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

// Declare the global ground detection structure
extern struct ground_detection_t global_ground_data;

// Function declarations for the ground segmentation module
void ground_segmentation_init(void); // Initialization function
struct image_t *ground_segmentation(struct image_t *img); // Main processing function
void ground_segmentation_periodic(void); // Periodic function for data handling

#endif /* CV_GROUND_SEG_H */