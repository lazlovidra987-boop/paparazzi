/*
 * Copyright (C) Roland Meertens
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/orange_avoider/orange_avoider.h"
 * @author Roland Meertens
 * Example on how to use the colours detected to avoid orange pole in the cyberzoo
 */

#ifndef ORANGE_AVOIDER_H
#define ORANGE_AVOIDER_H

#include <stdint.h>  
// settings
extern float oa_color_count_frac;
// Añadir junto a las otras extern:
extern int32_t edge_blocks_detected;

// functions
extern void orange_avoider_init(void);
extern void orange_avoider_periodic(void);

#endif

