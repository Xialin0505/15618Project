#include <iostream>
#include <unistd.h>
#include <fstream>

#define RAND_SEED 1234
#define MAX_WEIGHT 100000

// parameter
std::string output_filename;
int vertex_number;
int density;
int start;

// graph
int graph_size;
float* graph;
int* edge_count;
int int_array;

void init() {
    srand(RAND_SEED);

    graph_size = vertex_number * vertex_number * sizeof(float);
    int_array = vertex_number * sizeof(int); 
    graph = (float*)malloc(graph_size * sizeof(float));
    edge_count = (int*)malloc(int_array);

    start = (rand() % vertex_number);
}

void construct_graph() {
    int i;                  
    int rand_vertex;        
    int curr_num_edges;     
    float weight;    

    printf("Initializing a connected graph...");
    for (i = 1; i < vertex_number; i++) {
        rand_vertex = (rand() % i);                     
        weight = (rand() % MAX_WEIGHT) + 1;             
        graph[rand_vertex * vertex_number + i] = weight;   
        graph[i * vertex_number + rand_vertex] = weight;
        edge_count[i] += 1;                     
        edge_count[rand_vertex] += 1;
    }
    printf("done!\n");

    printf("Checking density...");
    for (i = 0; i < vertex_number; i++) {    
        curr_num_edges = edge_count[i];         
        while (curr_num_edges < density) {      
            rand_vertex = (rand() % vertex_number);  
            weight = (rand() % MAX_WEIGHT) + 1;     
            if ((rand_vertex != i) && (graph[i * vertex_number + rand_vertex] == 0)) { 
                graph[i * vertex_number + rand_vertex] = weight;
                graph[rand_vertex * vertex_number + i] = weight;
                edge_count[i] += 1;
                curr_num_edges ++;              
            }
        }
    }
    printf("done!\n");
}

void write_graph() {
    std::ofstream out_file(output_filename, std::fstream::out);
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

int main(int argc, char *argv[]) {
    int opt;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " -f output_filename -v vertex_number -d density\n";
        exit(1);
    }

    while ((opt = getopt(argc, argv, "f:v:d:")) != -1) {
        switch (opt) {
        case 'f':
            output_filename = optarg;
            break;
        case 'v':
            vertex_number = atof(optarg);
            break;
        case 'd':
            density = atoi(optarg);
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " -f output_filename -v vertex_number -d density\n";
        }
    }

    // printf("%s, %d, %d\n", output_filename.c_str(), vertex_number, density);
    

    init();
    construct_graph();
    write_graph();

    return 0;
}