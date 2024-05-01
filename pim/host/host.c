#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <dpu.h>
#include <getopt.h>
#include <common.h>

#include <timer.h>

#define FILENAME_LEN 50
#ifndef DPU_BINARY
#define DPU_BINARY "bin/dijkstra_dpu"
#endif

int vertex_number;
int* dist;
int* visited;
int* parent;
int64_t* graph;
int start;
char output_file[FILENAME_LEN];
char input_file[FILENAME_LEN];

int graph_size;
int int_array;
int data_array;

void construct_graph() {
    FILE* f = fopen(input_file, "r");

    char firstline[100];
    fgets(firstline, 100, f);

    char v[100];
    char vstart[100];
    int i = 0;
    for (; i < 100; i++) {
        if (firstline[i] == ' ') {
            break;
        }
        v[i] = firstline[i];
    }
    v[i] = '\0';
    i++;

    int idx = 0;
    for (; i < 100; i++) {
        if (firstline[i] == '\n') break;
        vstart[idx] = firstline[i];
        idx ++;
    }
    vstart[idx] = '\0';

    vertex_number = atoi(v);
    start = atoi(vstart);

    graph_size = vertex_number * vertex_number * sizeof(int64_t);
    int_array = vertex_number * sizeof(int);
    data_array = vertex_number * sizeof(int);

    graph = (int64_t*)malloc(graph_size);
    dist = (int*)malloc(data_array);
    parent = (int*)malloc(int_array);
    visited = (int*)malloc(int_array);

    idx = 0;
    char buf[100];

    for (int i = 0; i < vertex_number; i++) {
        for (int j = 0; j < vertex_number; j++) {
            while (1) {
                buf[idx] = fgetc(f);
                if (buf[idx] == ' ') break;
                idx ++;
            }
            buf[idx] = '\0';

            graph[i * vertex_number + j] = (int64_t)atoi(buf);

            idx = 0;
        }
    }

    fclose(f);
}

void write_graph() {
    FILE* f = fopen("output.txt", "w");
    char buf[100];
    sprintf(buf, "%d %d\n", vertex_number, start);
    fprintf(f, buf);

    for (int i = 0; i < vertex_number; i++) {
        for (int j = 0; j < vertex_number; j++) {
            int v = (int64_t)graph[i * vertex_number + j];
            sprintf(buf, "%d ", v);
            fprintf(f, buf);
        }
        fprintf(f, "\n");
    }

    fclose(f);
}

void write_output() {
    FILE* f = fopen(output_file, "w");
    char buf[100];

    for (int i = 0; i < vertex_number; i++) {
        int v = (int64_t)dist[i];
        sprintf(buf, "%d\n", v);
        fprintf(f, buf);
    }

    fclose(f);
}

void clean() {
    free(graph);
    free(dist);
    free(parent);
    free(visited);
}

void printGraph() {
    for (int i = 0; i < vertex_number; i++) {
        for (int j = 0; j < vertex_number; j++) {
            printf("%ld ", graph[i * vertex_number + j]);
        }
        printf("\n");
    }
}

void dijkstra() {
    int nr_of_dpus;
    struct dpu_set_t dpu_set, dpu;
    dpu_results_t results[NR_DPUS];

    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));

    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
    printf("Allocated %d DPU(s)\n", nr_of_dpus);

    printf("Load input data\n");
    DPU_FOREACH(dpu_set, dpu)
	{
		DPU_ASSERT(dpu_prepare_xfer(dpu, graph));
	}
    
	DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, graph_size, DPU_XFER_DEFAULT));
    //DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "graph", 0, graph_size, DPU_XFER_DEFAULT));

    printf("Load vertex\n");
    int interval = vertex_number / NR_DPUS;
    int eachdpu;
    dpu_arg args[NR_DPUS];
    DPU_FOREACH(dpu_set, dpu, eachdpu)
	{
        args[eachdpu].startidx = eachdpu * interval;
        args[eachdpu].endidx = (eachdpu + 1) * interval;
        args[eachdpu].num_vertex = vertex_number;
        args[eachdpu].graph_size = graph_size;
        args[eachdpu].start_node = start;
        DPU_ASSERT(dpu_prepare_xfer(dpu, &args[eachdpu]));
	}

    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_ARG", 0, sizeof(dpu_arg), DPU_XFER_DEFAULT));

    struct timespec start_time, end_time, delta;

    clock_gettime(CLOCK_REALTIME, &start_time);
    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    clock_gettime(CLOCK_REALTIME, &end_time);
    
    delta = diff(start_time, end_time); 
    printf("execution time: %d.%.9ld s\n", (int)delta.tv_sec, delta.tv_nsec);

    DPU_FOREACH (dpu_set, dpu) {
        DPU_ASSERT(dpu_log_read(dpu, stdout));
    }

    uint32_t each_dpu;
    DPU_FOREACH (dpu_set, dpu, each_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &results[each_dpu]));
    }

    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "DPU_RESULTS", 0, sizeof(dpu_results_t), DPU_XFER_DEFAULT));

    int idx = 0;
    DPU_FOREACH (dpu_set, dpu, each_dpu) {
        for (unsigned int each_tasklet = 0; each_tasklet < NR_TASKLETS; each_tasklet++) {
            dpu_result_t *result = &results[each_dpu].tasklet_result[each_tasklet];
            
            for (int i = 0; i < VERTEX_NUM_EACH_TASKLET; i++){
                dist[idx] = result->pathlength[i];
                idx ++;
                //printf("%d\n", result->pathlength[i]);
            }
        }
    }

    DPU_ASSERT(dpu_free(dpu_set));
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: -f input_filename -o output_filename\n");
        exit(1);
    }

    int opt;
    while ((opt = getopt(argc, argv, "f:o:")) != -1) {
        switch (opt) {
        case 'f':
            snprintf(input_file, FILENAME_LEN, "%s", optarg);
            break;
        case 'o':
            snprintf(output_file, FILENAME_LEN, "%s", optarg);
            break;
        default:
            printf("Usage: -f input_filename -o output_filename\n");
            exit(1);
        }
    }

    construct_graph();
    dijkstra();
    write_output();
    //write_graph();

    clean();
    return 0;
}