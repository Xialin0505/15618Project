// reference: https://github.com/srrcboy/dijkstra-CUDA/blob/master/dijkstra_cuda.cu

#include <cuda.h>
#include <stdio.h>  
#include <stdlib.h>
#include <time.h>
#include <math.h>

//Parameters; modify as needed
#define VERTICES 16384           //number of vertices
#define DENSITY 16              //minimum number of edges per vertex. DO NOT SET TO >= VERTICES
#define MAX_WEIGHT 100000      //max edge length + 1
#define INF_DIST 1000000000     //"infinity" initial value of each node
#define CPU_IMP 1               //number of Dijkstra implementations (non-GPU)
#define GPU_IMP 1               //number of Dijkstra implementations (GPU)
#define THREADS 2               //number of OMP threads
#define RAND_SEED 1234          //random seed
#define THREADS_BLOCK 512


__global__ void closestNodeCUDA(float* node_dist, int* visited_node, int* global_closest, int num_vertices) {
    float dist = INF_DIST + 1;
    int node = -1;
    int i;

    for (i = 0; i < num_vertices; i++) {
        if ((node_dist[i] < dist) && (visited_node[i] != 1)) {
            dist = node_dist[i];
            node = i;
        }
    }

    global_closest[0] = node;
    visited_node[node] = 1;
}

__global__ void cudaRelax(float* graph, float* node_dist, int* parent_node, int* visited_node, int* global_closest) {
    int next = blockIdx.x*blockDim.x + threadIdx.x;    //global ID
    int source = global_closest[0];

    float edge = graph[source*VERTICES + next];
    float new_dist = node_dist[source] + edge;

    if ((edge != 0) &&
        (visited_node[next] != 1) &&
        (new_dist < node_dist[next])) {
        node_dist[next] = new_dist;
        parent_node[next] = source;
    }

}

void init_graph() {

}

int main() {
    init_graph();


    return 0;
}