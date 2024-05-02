#include <iostream>
#include <unistd.h>
#include <fstream>
#include <time.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <x86intrin.h>

#define INFINITE_DIST 1000000000

int vertex_number;
int* dist;
int* visited;
int* parent;
int* graph;
int start;
std::string input_file;
std::string output_file;

int graph_size;
int int_array;
int data_array;

void setIntArrayValue(int* in_array, int array_size, int init_value);
void setDataArrayValue(int* in_array, int array_size, int init_value);
int closestNode(int* node_dist, int* visited_node, int num_vertices);

int closestNode(int* node_dist, int* visited_node) {
    int distance = INFINITE_DIST + 1;    //set start to infinity+1, so guaranteed to pull out at least one node
    int node = -1;              //closest non-visited node
    int i;                      //iterator

    for (i = 0; i < vertex_number; i++) {
        if ((node_dist[i] < distance) && (visited_node[i] == 0)) {  //if closer and not visited
            node = i;               //select node
            distance = node_dist[i];    //new closest distance
        }
    }
    return node;    //return closest node
}

void dijkstraCPUSerial(int* graph, int* node_dist, int* parent_node, int* visited_node, int v_start) {

    //reset/clear data from previous runs
    setDataArrayValue(node_dist, vertex_number, INFINITE_DIST);     //all node distances are infinity
    setIntArrayValue(parent_node, vertex_number, -1);          //parent nodes are -1 (no parents yet)
    setIntArrayValue(visited_node, vertex_number, 0);          //no nodes have been visited
    node_dist[v_start] = 0;                     //start distance is zero; ensures it will be first pulled out

    int i, next;
    for (i = 0; i < vertex_number; i++) {
        int curr_node = closestNode(node_dist, visited_node); //get closest node not visited
        visited_node[curr_node] = 1;                                        //set node retrieved as visited

        for (next = 0; next < vertex_number; next++) {
            int new_dist = node_dist[curr_node] + graph[curr_node * vertex_number + next];
            
            if ((visited_node[next] != 1)
                && (graph[curr_node * vertex_number + next] != (int)(0))
                && (new_dist < node_dist[next])) {
                node_dist[next] = new_dist;        //update distance
                parent_node[next] = curr_node;     //update predecessor
            }
        }
    }
}

void setIntArrayValue(int* in_array, int array_size, int init_value) {
    int i;
    for (i = 0; i < array_size; i++) {
        in_array[i] = init_value;
    }
}

/*  Initialize elements of a 1D data_t array with an initial value   */
void setDataArrayValue(int* in_array, int array_size, int init_value) {
    int i;
    for (i = 0; i < array_size; i++) {
        in_array[i] = init_value;
    }
}

/*  Construct graph with no edges or weights     */
void initializeGraphZero(int* graph, int num_vertices) {
    int i, j;

    for (i = 0; i < num_vertices; i++) {
        for (j = 0; j < num_vertices; j++) {           //weight of all edges initialized to 0
            graph[i * num_vertices + j] = (int)0;
        }
    }
}

void contructGraph() {
    std::ifstream fin(input_file);
    fin >> vertex_number >> start; 
    printf("%s: vertex %d, start %d\n", input_file.c_str(), vertex_number, start);

    graph_size = vertex_number * vertex_number * sizeof(int);
    int_array = vertex_number * sizeof(int);
    data_array = vertex_number * sizeof(int);

    graph = (int*)malloc(graph_size);
    dist = (int*)malloc(data_array);
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
            int weight = graph[i * vertex_number + j];
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

void clean() {
    free(graph);
    free(dist);
    free(parent);
    free(visited);
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

    struct timespec start_time, end_time, delta;

    uint64_t start_cycle = _rdtsc();
    clock_gettime(CLOCK_REALTIME, &start_time);
    dijkstraCPUSerial(graph, dist, parent, visited, start);
    clock_gettime(CLOCK_REALTIME, &end_time);
    uint64_t end_cycle = _rdtsc();

    double cycles = (double)(end_cycle - start_cycle);
	printf("Total CPU cycles : %g\n", cycles);
    
    delta = diff(start_time, end_time); 
    printf("execution time: %d.%.9ld s\n", (int)delta.tv_sec, delta.tv_nsec);

    write_output();
    clean();
}