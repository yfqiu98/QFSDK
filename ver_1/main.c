#include <stdio.h>
#include <stdlib.h>
#include "ffbpn.h"

int main()
{

    int input_node_num = 36;
    int hidden_layer_num = 4;
    int hidden_node_num = 3;
    int output_node_num = 4;
    double learning_rate = 0.001;
    double neuron_bias = 0.1;
    int epoch = 50;
    int data_len = 744;
    FILE *trainFile = fopen("Data_4_mix.csv", "r");

    ffbpn(input_node_num, hidden_node_num, hidden_layer_num, output_node_num, learning_rate, neuron_bias, trainFile, data_len, epoch);

    return 0;
}




