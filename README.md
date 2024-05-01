# 15618 Project
### How to run

#### Data
The data folder contain all the graph test file, with a graph generator to generate random graph, and store in a file.
The generator run as the following
```
./generator -f output_filename -v vertex_number -d density
```

#### Baseline
The baseline is CPU serial version of the Dijkstra algorithm, as a baseline to compare with the GPU and PIM version.
The baseline run as the following
```
./baseline -f input_filename -o output_filename
```

This program take a input graph file located in the `data` folder, and the output file name to write the minimum distance between the source vertex and the rest of the vertex.

#### GPU
The gpu is the GPU version of the Dijkstra algorithm.
The gpu is run as the following
```
./gpu -f input_filename -o output_filename
```

The program take a input graph file located in the `data` folder, and the output file name to write the minimum distance between the source vertex and the rest of the vertex.

The output file generated can be tested against the serial version to check the correctness using `diff`.

### PIM
The macro `VERTEX_NUM_EACH_TASKLET` in `commmon.h` must match the input vertex number to get the correct result
Due to the small stack and heap size of each DPU, DPU is unable to process vertex number more than 1024.

The pim is run as the following
```
./bin/dijkstra_host -f input_filename -o output_filename
```

Note that the pim code have to be run in the directory.