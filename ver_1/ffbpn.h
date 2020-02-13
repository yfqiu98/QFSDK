#ifndef _FFBPN_h
#define _FFBPN_h


#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_LINE_SIZE 10000

//**********   Decleration Array  ***********//

double Oscillator[1001][100];
double Oscillator_Derivation[1000];



typedef struct Node{
    double states;
    double error;
    double output;
    double *weight;
}neuron;


typedef struct InputNode
{
    double output;
    double error;
}inputNeuron;


typedef struct hiddenNode
{
    double states;
    double error;
    double *weight;
    double output;
}hiddenNeuron;


typedef struct outputNode
{
    double states;
    double error;
    double *weight;
    double output;
}outputNeuron;

void ffbpn(
    int input_node_num, int hidden_node_num, int hidden_layer_num, int output_node_num,
    double learningRate, double bias,
    FILE *trainData, int dataLen,
    int epochTime
);

typedef struct nerualNetwork
{
    int input_node_num;
    int hidden_layer_num;
    int hidden_node_num;
    int output_node_num;
    double learningRate;
    double bias;
    neuron *input_layer;
    neuron *hidden_layer;
    neuron *output_layer;
}nn;

double * prepareData(nn *ann, int dataType, double *tradingData,int tradingData_size, int tradingData_length, double *qpl, int qpl_size, int slideWindowSize);



#endif