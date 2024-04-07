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