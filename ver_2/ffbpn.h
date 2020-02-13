#ifndef _FFBPN_h
#define _FFBPN_h

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>


/**
 * Description: define basic neuron structure
 * 
 * inputState: the sum of pervoius layer output
 * weight: each pervoius neurons weight
 * output: output value after activation function
 * error: for bp part record error and learn
*/
typedef struct neuron{
    double inputState;
    double *weight;
    double output;
    double error;
    struct neuron *pervious;
    struct neuron *next;
}neuron;

typedef struct inputLayer{
    int activationFunctionType;
    double bias;
    neuron *neuronList;
    int neuronNum;
}inputLayer;

typedef struct hiddenLayer{
    int activationFunctionType;
    double bias;
    neuron *neuronList;
    int neuronNum;
}hiddenLayer;

typedef struct outputLayer{
    int activationFunctionType;
    neuron *neuronList;
    int neuronNum;
}outputLayer;




#endif