#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <mram.h>
#include <barrier.h>
#include <perfcounter.h>
#include "common.h"

__host dpu_results_t DPU_RESULTS;
__host dpu_arg DPU_ARG;

int closestNode(int* node_dist, int* visited_node) {
    int distance = INFINITE_DIST + 1;  
    int node = -1;         
    int i;                      

    for (i = 0; i < DPU_ARG.num_vertex; i++) {
        if ((node_dist[i] < distance) && (visited_node[i] == 0)) {  //if closer and not visited
            node = i;               //select node
            distance = node_dist[i];    //new closest distance
        }
    }
    return node;    //return closest node
}

void printGraph(uint32_t graph) {
    int64_t cache;
    
    for (int i = 0; i < DPU_ARG.num_vertex; i++) {
        for (int j = 0; j < DPU_ARG.num_vertex; j++) {
            uint32_t addr = graph + (i * DPU_ARG.num_vertex + j) * sizeof(int64_t);
            mram_read((__mram_ptr void const *) addr, &cache, BLOCK_SIZE);
            printf("%ld ", cache);
        }
        printf("\n");
    }
}

int main() {
    uint32_t tasklet_id = me();
    dpu_result_t *result = &DPU_RESULTS.tasklet_result[tasklet_id];
    uint32_t start = DPU_ARG.startidx + tasklet_id * VERTEX_NUM_EACH_TASKLET;
    
    uint32_t graph = (uint32_t) DPU_MRAM_HEAP_POINTER;
    int *node_dist = (int *)mem_alloc(DPU_ARG.num_vertex * sizeof(int));
    int *visited_node = (int *)mem_alloc(DPU_ARG.num_vertex * sizeof(int));
    int64_t cost;
 
    for (int i = 0; i < DPU_ARG.num_vertex; i++) {
        node_dist[i] = INFINITE_DIST;
        visited_node[i] = 0;
    }
    node_dist[DPU_ARG.start_node] = 0;

    int i, next;
    for (i = 0; i < DPU_ARG.num_vertex; i++) {
    
        int curr_node = closestNode(node_dist, visited_node);
        visited_node[curr_node] = 1;                                      

        for (next = 0; next < DPU_ARG.num_vertex; next++) {
            uint32_t addr = graph + (curr_node * DPU_ARG.num_vertex + next) * sizeof(int64_t);
            mram_read((__mram_ptr void const *) addr, &cost, BLOCK_SIZE);
            
            int64_t new_dist = node_dist[curr_node] + cost;

            if ((visited_node[next] != 1)
                && (cost != (int64_t)(0))
                && (new_dist < node_dist[next])) {
                node_dist[next] = new_dist;        //update distance
            }
        }
    }
    
    for (int i = 0; i < VERTEX_NUM_EACH_TASKLET; i++) {
        result->pathlength[i] = node_dist[i];
    }

    return 0;
}