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
#define MAX_WEIGHT 100000      
#define INF_DIST 1000000000     
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
    extern __shared__ float shared_dist[];
    int thread_id = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    float min_dist = INF_DIST;
    int min_node = -1;

    // Load to shared memory and initialize
    if (global_id < num_vertices) {
        if (!visited_node[global_id] && node_dist[global_id] < min_dist) {
            min_dist = node_dist[global_id];
            min_node = global_id;
        }
    }
    shared_dist[thread_id] = min_dist;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_id < s && global_id + s < num_vertices) {
            if (shared_dist[thread_id + s] < shared_dist[thread_id]) {
                shared_dist[thread_id] = shared_dist[thread_id + s];
                min_node = thread_id + s;
            }
        }
        __syncthreads();
    }

    // Write result to global memory
    if (thread_id == 0) {
        global_closest[0] = min_node;
        visited_node[min_node] = 1;
    }
}

__global__ void cudaRelax(float* graph, float* node_dist, int* parent_node, int* visited_node, int* global_closest, int num_vertices) {
    int next = blockIdx.x * blockDim.x + threadIdx.x;    //global ID
    int source = global_closest[0];

    float edge = graph[source * num_vertices + next];
    float new_dist = node_dist[source] + edge;

    if ((edge != 0) &&
        (visited_node[next] != 1) &&
        (new_dist < node_dist[next])) {
        node_dist[next] = new_dist;
        parent_node[next] = source;
    }

}

void setIntArrayValue(int* in_array, int array_size, int init_value) {
    int i;
    for (i = 0; i < array_size; i++) {
        in_array[i] = init_value;
    }
}

/*  Initialize elements of a 1D data_t array with an initial value   */
void setDataArrayValue(float* in_array, int array_size, float init_value) {
    int i;
    for (i = 0; i < array_size; i++) {
        in_array[i] = init_value;
    }
}

/*  Construct graph with no edges or weights     */
void initializeGraphZero(float* graph, int num_vertices) {
    int i, j;

    for (i = 0; i < num_vertices; i++) {
        for (j = 0; j < num_vertices; j++) {           //weight of all edges initialized to 0
            graph[i * num_vertices + j] = (float)0;
        }
    }
}

void dijkstra() {
    cudaEvent_t exec_start, exec_stop;              //timer for execution only
    float elapsed_exec;                             //elapsed time
    cudaEventCreate(&exec_start);
    cudaEventCreate(&exec_stop);

    float* gpu_graph;
    float* gpu_dist;
    int* gpu_parent;
    int* gpu_visited;
    cudaMalloc((void**)&gpu_graph, graph_size);
    cudaMalloc((void**)&gpu_dist, data_array);
    cudaMalloc((void**)&gpu_parent, int_array);
    cudaMalloc((void**)&gpu_visited, int_array);

    int* closest = (int*)malloc(sizeof(int));
    *closest = -1;
    int* gpu_closest;
    cudaMalloc((void**)&gpu_closest, sizeof(int));
    cudaMemcpy(gpu_closest, closest, sizeof(int), cudaMemcpyHostToDevice);

    setDataArrayValue(dist, vertex_number, INF_DIST);
    setIntArrayValue(parent, vertex_number, -1);
    setIntArrayValue(visited, vertex_number, 0);
    dist[start] = 0;

    cudaMemcpy(gpu_graph, graph, graph_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_dist, dist, data_array, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_parent, parent, int_array, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_visited, visited, int_array, cudaMemcpyHostToDevice);

    dim3 blockClosest(THREADS_BLOCK, 1, 1); // Use full blocks for reduction
    dim3 gridClosest((vertex_number + THREADS_BLOCK - 1) / THREADS_BLOCK, 1, 1); // Cover all vertices

    dim3 gridRelax(vertex_number / THREADS_BLOCK, 1, 1);
    dim3 blockRelax(THREADS_BLOCK, 1, 1);

    cudaEventRecord(exec_start);
    for (int i = 0; i < vertex_number; i++) {
        closestNodeCUDA<<<gridClosest, blockClosest, THREADS_BLOCK * sizeof(float)>>>(gpu_dist, gpu_visited, gpu_closest, vertex_number);
        cudaRelax<<<gridRelax, blockRelax>>>(gpu_graph, gpu_dist, gpu_parent, gpu_visited, gpu_closest, vertex_number);
    }
    cudaEventRecord(exec_stop);

    cudaMemcpy(dist, gpu_dist, data_array, cudaMemcpyDeviceToHost);
    cudaMemcpy(parent, gpu_parent, int_array, cudaMemcpyDeviceToHost);
    cudaMemcpy(visited, gpu_visited, int_array, cudaMemcpyDeviceToHost);

    cudaFree(gpu_graph);
    cudaFree(gpu_dist);
    cudaFree(gpu_parent);
    cudaFree(gpu_visited);

    cudaEventElapsedTime(&elapsed_exec, exec_start, exec_stop);
    printf("\n\nCUDA Time (ms): %7.9f\n", elapsed_exec);
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

void clean() {
    free(graph);
    free(dist);
    free(parent);
    free(visited);
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

struct timespec diff(struct timespec start, struct timespec end)
{
    struct timespec temp;
    if ((end.tv_nsec - start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
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
    dijkstra();
    write_output();
    clean();
    return 0;
}