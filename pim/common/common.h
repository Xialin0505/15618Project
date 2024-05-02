#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "stdlib.h"

#define BLOCK_SIZE sizeof(int64_t)

#define XSTR(x) STR(x)
#define STR(x) #x

#define DPU_BUFFER dpu_mram_buffer
#define DPU_CACHES dpu_wram_caches

#define VERTEX_NUM_EACH_TASKLET 1024
#define INFINITE_DIST 1000000000

typedef struct {
    uint32_t pathlength[VERTEX_NUM_EACH_TASKLET];
    uint32_t cycles;
} dpu_result_t;

typedef struct {
    dpu_result_t tasklet_result[NR_TASKLETS];
} dpu_results_t;

typedef struct {
    uint32_t startidx;
    uint32_t endidx;
    uint32_t num_vertex;
    uint32_t graph_size;
    int start_node;
} dpu_arg;