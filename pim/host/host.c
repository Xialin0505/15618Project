#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>

#define FILENAME_LEN 50

int vertex_number;
float* dist;
int* visited;
int* parent;
float* graph;
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

    graph_size = vertex_number * vertex_number * sizeof(float);
    int_array = vertex_number * sizeof(int);
    data_array = vertex_number * sizeof(float);

    graph = (float*)malloc(graph_size);
    dist = (float*)malloc(data_array);
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

            graph[i * vertex_number + j] = (float)atoi(buf);
            printf("%f\n", graph[i * vertex_number + j]);

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
            int v = (int)graph[i * vertex_number + j];
            sprintf(buf, "%d ", v);
            fprintf(f, buf);
        }
        fprintf(f, "\n");
    }

    fclose(f);
}

void clean() {
    free(graph);
    free(dist);
    free(parent);
    free(visited);
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
    write_graph();

    clean();
    return 0;
}