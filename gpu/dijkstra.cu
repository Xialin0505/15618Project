// reference: https://github.com/srrcboy/dijkstra-CUDA/blob/master/dijkstra_cuda.cu

#include <cuda.h>
#include <stdio.h>  
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <unistd.h>
#include <fstream>

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


int vertex_number;
float* dist;
int* visited;
int* parent;
float* graph;
int start;
std::string input_file;
std::string output_file;

int graph_size;
int int_array;
int data_array;


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

void contructGraph() {
    std::ifstream fin(input_file);
    fin >> vertex_number >> start; 
    printf("%s: vertex %d, start %d\n", input_file.c_str(), vertex_number, start);

    graph_size = vertex_number * vertex_number * sizeof(float);
    int_array = vertex_number * sizeof(int);
    data_array = vertex_number * sizeof(float);

    graph = (float*)malloc(graph_size);
    dist = (float*)malloc(data_array);
    parent = (int*)malloc(int_array);
    visited = (int*)malloc(int_array);

    for (int i = 0; i < vertex_number; i++) {
        for (int j = 0; j < vertex_number; j++) {
            fin >> graph[i * vertex_number + j];
        }
    }

    fin.close();
}

void write_graph() {
    std::ofstream out_file("test.txt", std::fstream::out);
    out_file << vertex_number << ' ' << start << '\n';

    for (int i = 0; i < vertex_number; i++) {
        for (int j = 0; j < vertex_number; j++) {
            float weight = graph[i * vertex_number + j];
            out_file << weight << ' ';
        }
        out_file << "\n";
    }

    out_file.close();
}

void write_output() {
    std::ofstream out_file(output_file, std::fstream::out);
    for (int i = 0; i < vertex_number; i++) {
        out_file << dist[i] << '\n';
    }

    out_file.close();
}

int main(int argc, char *argv[]) {

    int opt;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " -f input_filename -o output_filename\n";
        exit(1);
    }

    while ((opt = getopt(argc, argv, "f:o:")) != -1) {
        switch (opt) {
        case 'f':
            input_file = optarg;
            break;
        case 'o':
            output_file = optarg;
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " -f input_filename -o output_filename\n";
        }
    }

    contructGraph();
    write_graph();

    return 0;
}