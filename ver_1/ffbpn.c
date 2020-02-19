/*
    FeedForward Backpropagation Network
    Activation Function: Sigmod Function
    Numbers of hidden layer: 1

    Created by Henry
    Last modify: Jun 3, 2019
*/

#include "ffbpn.h"

//define the sigmod function
double sigmod_bin(double x){
    return 1.0 / (1.0 + exp(-1.0 * x));
}

bool Load_Oscillator(){
	FILE* fp = fopen("Lee_Oscillator", "r");

	if(fp == NULL){
		printf("Open File Failed\n");
		return false;
	}
	
	char line[MAX_LINE_SIZE];
    char *result = NULL;
	int row = 0;
	int column = 0;
	double tmp = 0.0;

	while(fgets(line, MAX_LINE_SIZE, fp) != NULL) {
		
		column = 0;
        result = strtok(line, ",");
		

        while(result != NULL) {
            // printf( "%.9f ", atof(result));
			tmp = atof(result);

			//printf("%d ============ %s\n",row, result);
			//break;
			// printf("%.9f ", tmp);
			Oscillator[row][column] = tmp;
            result = strtok(NULL, ",");
			//i++;
			column++;
        }
		row++;
		//printf("\n row : %d", row);
		
    }
    
    fclose(fp);
}

bool Load_Oscillator_Derivation(){
	FILE* fp = fopen("Oscillator_Derivation", "r");

	if(fp == NULL){
		printf("Open File Failed\n");
		return false;
	}
	
	char line[MAX_LINE_SIZE];
    char *result = NULL;
	int row = 0;
	
	double tmp = 0.0;

	while(fgets(line, MAX_LINE_SIZE, fp) != NULL) {

		tmp = atof(line);

		// printf("%.9f ", tmp);
		Oscillator_Derivation[row] = tmp;
  
		row++;
		
    }

    fclose(fp);
}

////  This function is used to replace the activation function

double Oscillator_activation_func(double input){

	int rowIndex = input * 1000;
	int columnIndex = 0;

	if(rowIndex <= 0)
		return 0.0000372;
	else if(rowIndex > 1000)
	    return 0.999972831;

	srand(time(NULL));
	columnIndex = rand()%100;
	return Oscillator[rowIndex][columnIndex];
}


////  This function is used to replace the derivation of the activation function

double Oscillator_Derivation_func(double input){

	int rowIndex = input * 1000;

	if(rowIndex <= 0)
		return 0.00000075;
	else if(rowIndex > 1000)
	    return -0.00000075;

	return Oscillator_Derivation[rowIndex];
}

///////////////////////////////////////////////


//print the value in the NN model in order to check the value
void printNeuron(neuron input, int weightSize)
{
    printf("\nStates = %.20lf, Output = %.20lf, Error = %.20lf Weight:", input.states, input.output, input.error);
    for(int i = 0; i < weightSize; i++)
    {
        printf("\t%.20lf", input.weight[i]);
    }
    printf("\n");
}

void checkHidden(neuron *hidden_layer, int input_node_num, int hidden_node_num, int hidden_layer_num)
{
    for(int i = 0; i < hidden_layer_num*hidden_node_num; i++)
    {
        if (i < hidden_node_num)
        {
            printNeuron(hidden_layer[i], input_node_num);
        }
        else
        {
            printNeuron(hidden_layer[i], hidden_node_num);
        }
    }
}

/**
Description:
    activation function in for neuron calculate the output from states
Parameter:
    x:  Input value
    type:   activation function type
Return:
    result:     double
Activation Function Type:
    1:  sigmoid
    default:    sigmod

*/
double activationFunction(double x, int activation_type)
{
    double result;

    switch (activation_type)
    {
    case 1:
        result = 1.0 / (1.0 + exp(-1.0 * x));
        break;

    case 2:
        result = Oscillator_activation_func(x);

    //the default activation function is sigmoid function
    default:
        result = 1.0 / (1.0 + exp(-1.0 * x));
        break;
    }

    return result;
}

//define random number function for initialize
double random_0_1(){
    //srand(time(NULL));
    double number = (double)rand()/(double)RAND_MAX;
    return number;
}

//create array for weight array
double * create_array(int arraySize)
{
    double *array = (double *)malloc(sizeof(double) * arraySize);
    if(!array)
    {
        printf("Opp! Out of memory!\n");
        exit(1);
    }
    return array;
}

void initInputlayer(neuron *inputLayer, int arraySize)
{
    for (int i = 0; i < arraySize; i++)
    {
        inputLayer[i].states = 0;
        inputLayer[i].output = 0;
        inputLayer[i].error = 0;
    }
}

void initHiddenLayer(neuron *hiddenLayer, int input_node_num, int hidden_layer_num, int hidden_node_num)
{
    //initialize the random number seed
    srand(time(NULL));

    for(int i = 1; i <= hidden_layer_num; i++)
    {

        //the first hidden layer (input layer --> hidden layer)
        if(i == 1)
        {

            for(int j = 0; j < hidden_node_num; j++)
            {

                //initialize normal value
                hiddenLayer[j].output = 0;
                hiddenLayer[j].error = 0;
                hiddenLayer[j].states = 0;

                //create number of weight for layer
                hiddenLayer[j].weight = create_array(input_node_num);

                //random the weight value of layer
                for(int m = 0; m < input_node_num; m++)
                {
                    hiddenLayer[j].weight[m] = random_0_1();
                }
            }

        }


        //for normal hidden layer
        else
        {
            for(int j = (i-1)*hidden_node_num; j < i*hidden_node_num; j++)
            {

                //initialize normal value
                hiddenLayer[j].output = 0;
                hiddenLayer[j].error = 0;
                hiddenLayer[j].states = 0;

                //create number of weight for layer
                hiddenLayer[j].weight = create_array(hidden_node_num);

                //random the weight of the NN model
                for(int m = 0; m < hidden_node_num; m++)
                {
                    hiddenLayer[j].weight[m] = random_0_1();
                }
            }
        }

    }

}

void initOutputLayer(neuron *outputLayer, int arraySize, int weightSize)
{
    srand(time(NULL));

    for (int i = 0; i < arraySize; i++)
    {
        outputLayer[i].weight = create_array(weightSize + 1);
        outputLayer[i].error = 0;
        outputLayer[i].states = 0;
        outputLayer[i].output = 0;

        //initialize weight value in each outputNeuron
        for (int j = 0; j < weightSize + 1; j++)
        {
            outputLayer[i].weight[j] = random_0_1();
        }
    }
}


void loadInputLayer(neuron *input_layer, int input_node_num, double *input)
{
    for(int i = 0; i < input_node_num; i++)
    {
        input_layer[i].output = input[i];
    }
}


/*
Description:
    Calculate each neuron value by pervious layer and the weight
Parameter:
    *neuronNode: each neuron inside the Neural Network
    inputLayer: pervious layer for current neuron node calculate the states by the weight (not include bias)
    inputLayerSize: the size of pervious layer
    bias: the bias of each Neural Network
Return:
    void
 */
void feedforward_neuron(neuron *neuronNode, neuron inputLayer[], int inputLayerSize, double bias, int activation_type)
{
    //refresh the pervious neuron states for current time calculate
    neuronNode->states = 0.0;

    //for pervious layer calculate current neuron states
    for(int i = 0; i < inputLayerSize; i++)
    {
        neuronNode->states += inputLayer[i].output * neuronNode->weight[i+1];
    }

    //calculate bias into neuron
    neuronNode->states += bias * neuronNode->weight[0];

    //calculate output by activationFunction
    neuronNode->output = activationFunction(neuronNode->states, activation_type);               
    
}

/**
 * Description:
 *  To dispatch the whole FeedForward process from different layer
 * Parameter:
 *  input_layer[]: the input layer
 *  hidden_layer[]: the hidden layer, with one single array but include all hidden layers neuron
 *  output_layer[]: the output layer
 *  input_node_num: the node numbers of input layer
 *  hidden_node_num: the node numbers of hidden layer
 *  output_node_num: the node numbers of output layer
 *  hidden_layer_num: the numbers of the layers in hidden layer
 *  bias: bias for each layer define by the user
 * Return:
 *  void
 */
void FeedForward(
    neuron input_layer[], neuron hidden_layer[], neuron output_layer[],
    int input_node_num, int hidden_node_num, int output_node_num,int hidden_layer_num,
    double bias, int activation_type
)
{

    //for the first hidden layer (input layer --> hidden layer)
    for(int m = 0; m < hidden_node_num; m++)
    {
        feedforward_neuron(&hidden_layer[m], input_layer, input_node_num, bias, activation_type);
        //pervious[m] = hidden_layer[m];
    }

    //for calculate between hidden layer (hidden layer --> hidden layer)
    for (int n = 1; n < hidden_layer_num; n++)
    {
        //initialize the tempArray for save pervious layer
        neuron *pervious = (neuron *)malloc(hidden_node_num * sizeof(neuron));

        //each time it will record the last layer for current input
        for(int j=(n-1)*hidden_node_num; j < (n+1)*hidden_node_num; j++)
        {
            pervious[j%hidden_node_num] = hidden_layer[j];
        }

        for(int q=n*hidden_node_num; q < (n+1)*hidden_node_num; q++)
        {
            feedforward_neuron(&hidden_layer[q], pervious, hidden_node_num, bias, activation_type);
        }

    }

    //for calculate between the last hidden layer and output layer (hidden layer --> output layer)
    //the first record the last hidden layer for output layer calculate
    int tempIndex = 0;
    neuron *pervious = (neuron *)malloc(hidden_node_num * sizeof(neuron));
    for(int index = (hidden_layer_num-1)*hidden_node_num; index < hidden_layer_num*hidden_node_num; index++)
    {
        pervious[tempIndex] = hidden_layer[index];
    }

    for(int o = 0; o < output_node_num; o++)
    {
        feedforward_neuron(&output_layer[o], pervious, hidden_node_num, bias, activation_type);
    }

}

/**
 * Description:
 *  Calculate each neuron value and adjust the weight between different layer, in backpropagation step
 * Parameter:
 *  *neuronNode: Current neuron pointer point to neuron which need execute
 *  pervious_layer: Provide pervious layer value for current neuron to adjust weight
 *  pervious_layer_size: For for loop to limit the calculate times
 *  y: The actucal y value for NN to calculate error and adjust the weight in Backpropagation
 *  learningRate: The learningRate use to control the size of learning step in each trainning
 *  bias: The bias for NN to adjust the weight of bias in each layer
 * Return:
 *  loss:               double (for main function to calculate MSE)
 */
double backpropagation_neuron_out(neuron *neuronNode, neuron pervious_layer[], int pervious_layer_size, double y, double learningRate, double bias, int activation_type)
{
    // Determine the acitvation and choose derivation calculation
    switch(activation_type)
    {
        // Sigmoid
        case 1: 
            neuronNode->error = neuronNode->output * (1 - neuronNode->output) * (y - neuronNode->output);
        // Oscillator derivation (Pending)
        case 2: 
            // neuronNode->error = neuronNode->output * (1 - neuronNode->output) * (y - neuronNode->output);
            neuronNode->error = Oscillator_Derivation_func(neuronNode->states) * (y - neuronNode->output);
        default:
            // Derivation for Sigmoid: calculate the error by (y - y_out)*(1 - y_out)*y_out
            neuronNode->error = neuronNode->output * (1 - neuronNode->output) * (y - neuronNode->output);
            break;
    }
    

    // Oscillator Derivation
    // neuronNode->error = neuronNode->output * (1 - neuronNode->output) * (y - neuronNode->output);

    //update the weight from pervious layer
    for(int i = 0; i < pervious_layer_size; i++)
    {
        neuronNode->weight[i+1] = neuronNode->weight[i+1] + learningRate * neuronNode->error * pervious_layer[i].output;
    }

    //upate bias weight by current error
    neuronNode->weight[0] += neuronNode->error * bias;

    //for loss function calculate MSE
    double loss = neuronNode->output - y;
    return loss * loss;
}

/**
 * Description:
 *  for each neuron process backpropagation step
 * Parameter:
 *  neuronNode:
 *  pervious_layer:
 *  back_layer:
 *  back_layer_size:
 *  back_layer_index:
 *  learningRate:
 *  bias: pervious_layer_size:
 * 
 * Return:
 *  NULL
 */
void backpropagation_neuron_hid(neuron *neuronNode, neuron pervious_layer[], int pervious_layer_size, neuron back_layer[], int back_layer_size, int back_layer_index, double learningRate, double bias)
{
    //initialize accumulated error
    double accumulated_error = 0.0;

    //summrize all the error from last layer (right of current layer)
    for(int i = 0; i < back_layer_size; i++)
    {
        accumulated_error += back_layer[i].error * back_layer[i].weight[back_layer_index];
    }

    //calculate current neuron error (activation function is sigmod function)
    neuronNode->error = neuronNode->output * (1 - neuronNode->output) * accumulated_error;

    //update current neuron weight for pervious layer (Need attention for code review)
    for(int j = 0; j < pervious_layer_size; j++)
    {
        neuronNode->weight[j+1] = neuronNode->weight[j+1] + learningRate * neuronNode->error * pervious_layer[j].output;
    }

    //update the bias weight
    neuronNode->weight[0] = neuronNode->weight[0] + learningRate * neuronNode->error * bias;
}

/**
 * Description:
 *  Backpropagation process main function, responsible for whole backpropagation scheduling
 * Parameter:
 *
 */
double backpropagation(
    neuron input_layer[], neuron hidden_layer[], neuron output_layer[], double y[],
    int input_node_num, int hidden_node_num, int output_node_num, int hidden_layer_num,
    double learningRate, double bias, int activation_type
)
{
    double loss = 0;

    //the output layer to actually value (output layer --> y)
    //load perivous layer
    neuron *pervious = (neuron *)malloc(hidden_node_num * sizeof(neuron));
    for(int i = (hidden_layer_num-1)*hidden_node_num; i < hidden_node_num*hidden_layer_num; i++)
    {
        pervious[i%hidden_node_num] = hidden_layer[i];
    }

    //for output layer neuron to calculate
    for(int m = 0; m < output_node_num; m++)
    {
        loss = loss + backpropagation_neuron_out(&output_layer[m], pervious, hidden_node_num, y[m], learningRate, bias, activation_type);
    }
    free(pervious);


    //for the hidden layer part
    for(int i = hidden_layer_num; i > 0; i--)
    {

        //for the last hidden layer to output layer (hidden layer <-- output layer)
        if(i == hidden_layer_num)
        {
            //load the perivous layer for backpropagation calculate
            neuron *pervious = (neuron *)malloc(hidden_node_num * sizeof(neuron));
            for(int q = (i-2)*hidden_node_num; q < (i-1)*hidden_node_num; q++)
            {
                pervious[q%hidden_node_num] = hidden_layer[q];
            }

            for(int m = (i-1)*hidden_node_num; m < i*hidden_node_num; m++)
            {
                backpropagation_neuron_hid(&hidden_layer[m], pervious, hidden_node_num, output_layer, output_node_num, m%hidden_node_num, learningRate, bias);
            }

            free(pervious);
        }

        //for the first hidden layer (input layer <-- hidden layer)
        else if(i == 1)
        {
            //record the back layer for the calculate
            neuron *back = (neuron *)malloc(hidden_node_num * sizeof(neuron));
            for(int q = i*hidden_node_num; q < (i+1)*hidden_node_num; q++)
            {
                back[q%hidden_node_num] = hidden_layer[q];
            }

            for(int m = 0; m < hidden_node_num; m++)
            {
                backpropagation_neuron_hid(&hidden_layer[m], input_layer, input_node_num, back, hidden_node_num, m%hidden_node_num, learningRate, bias);
            }

            free(back);
        }

        //for normal hidden layer (hidden layer <-- hidden layer)
        else
        {
            //prepared for record pervious layer and back layer
            neuron *pervious = (neuron *)malloc(hidden_node_num * sizeof(neuron));
            neuron *back = (neuron *)malloc(hidden_node_num * sizeof(neuron));

            //record pervious layer
            for(int m = (i-2)*hidden_node_num; m < (i-1)*hidden_node_num; m++)
            {
                pervious[m%hidden_node_num] = hidden_layer[m];
            }

            //record back layer
            for(int n = i*hidden_node_num; n < (i+1)*hidden_node_num; n++)
            {
                back[n%hidden_node_num] = hidden_layer[n];
            }

            //start hidden neuron calculate
            for(int j = (i-1)*hidden_node_num; j < i*hidden_node_num; j++)
            {
                backpropagation_neuron_hid(&hidden_layer[j], pervious, hidden_node_num, back, hidden_node_num, j%hidden_node_num, learningRate, bias);
            }

            free(pervious);
            free(back);
        }

    }

    //return the MSE for evaluate the model accurcy
    return 0.5*loss;

}

/**
 * Description:
 *  Load the trainning data from the csv file, the format define by user
 * Parameter:
 *
 */
double * loadData(FILE *trainFile, int input_node_num, int output_node_num, int dataLen, int *actLen)
{
    double *train = (double *)malloc((dataLen+1)*(input_node_num+output_node_num) * sizeof(double));
    if(train == NULL){
        printf("Out of memery while loading trainning data\n");
        return NULL;
    }

    int lens = (input_node_num+output_node_num)*10;
    char *line = (char *)malloc(lens * sizeof(char));
    char *result;
    int index = 0;

    while (fgets(line, lens, trainFile) != NULL)
    {
        result = strtok(line, ",");
        int k = 0;
        while(result != NULL)
        {
            train[index*(input_node_num+output_node_num)+k] = atof(result);
            result = strtok(NULL, ",");
            k++;
        }
        index++;
    }

    *actLen = index;

    free(line);
    return train;
}

/**
 * Description:
 * 
 */
void addTraining(nn *ann, double *input, double *output)
{

}

/**
 * Description:
 * 
 */
int compute(nn *ann, double *input)
{

}

/**
 * Description:
 *  
 * Parameter:
 * 
 */
double * prepareData(nn *ann,int dataType, double *tradingData,int tradingData_size, int tradingData_length, double *qpl, int qpl_size, int slideWindowSize)
{
    //record the input node numbers of ann
    ann->input_node_num = (tradingData_size+qpl_size) * (slideWindowSize - 1);
    ann->output_node_num = tradingData_size;

    //initialize value
    double *trainingData_input = (double *)malloc(tradingData_length * (tradingData_size + qpl_size) * sizeof(double));
    double *trainingData_output = (double *)malloc(tradingData_size * sizeof(double));

    //for loop in trading record to traversal
    for(int i = 0; i < tradingData_length; i++)
    {
        //check if it is out of limitation
        if(i > tradingData_length - slideWindowSize)
        {
            break;
        }

        //for each trading data, shift slideWindowSize time to make time serise training data
        for(int j = 0; j < slideWindowSize; j++)
        {
            //record the output
            if(j == slideWindowSize - 1)
            {
                for(int m = 0; m < tradingData_size; m++)
                {
                    trainingData_output[i * tradingData_size + m] = tradingData[(i + j) * tradingData_size + m];
                }
            }

            //record the input
            else
            {
                //the first to transfer trading record to training data
                for(int m = 0; m < tradingData_size; m++)
                {
                    trainingData_input[i * (tradingData_size + qpl_size) + m] = tradingData[(i + j) * tradingData_size + m];
                }

                //transfer the qpl to training data
                for(int n = 0; n < qpl_size; n++)
                {
                    trainingData_input[i * (tradingData_size + qpl_size) + tradingData_size + n] = qpl[n];
                }
            }
            
        }
    }


    //for addTraining part
    if(dataType == 0)
    {
        addTraining(ann, trainingData_input, trainingData_output);
    }

    //for compute part
    else if(dataType == 1)
    {
        compute(ann, trainingData_input);
    }

    return trainingData_input;
}


/**
 * Description: saving the ffbpn model for next time training
 */
void saveModel(
    neuron input_layer[], neuron hidden_layer[], neuron output_layer[],
    int input_node_num, int hidden_layer_num, int hidden_node_num, int output_node_num,
    double learningRate, double bias
)
{
    printf("Start saving the neural network model...\n");

    FILE *model_file = fopen("ffbpn.model", "w");

    //first record the basic structure of the NN model
    fprintf(model_file, "%d,%d,%d,%d,%f,%f\n", input_node_num, hidden_layer_num, hidden_node_num, output_node_num, learningRate, bias);
    //printf("%d,%d,%d,%d,%f,%f\n", input_node_num, hidden_layer_num, hidden_node_num, output_node_num, learningRate, bias);

    printf("Saving hidden layer...");

    //start to record the hidden layer of the neural Network
    for(int i = 0; i < hidden_layer_num*hidden_node_num; i++)
    {
        //record states, output and error of each neuron
        fprintf(model_file, "%f,%f,%f", hidden_layer[i].states, hidden_layer[i].output, hidden_layer[i].error);
        //printf("%e,%e,%e", hidden_layer[i].states, hidden_layer[i].output, hidden_layer[i].error);

        //record all the weight of current neuron
        //for the first layer of the hidden layer
        if(i < hidden_node_num)
        {
            for(int j = 0; j < input_node_num; j++)
            {
                fprintf(model_file, ",%f", hidden_layer[i].weight[j]);
                //printf(",%e", hidden_layer[i].weight[j]);
            }
        }
        else
        {
            for(int j = 0; j < hidden_node_num; j++)
            {
                fprintf(model_file, ",%f", hidden_layer[i].weight[j]);
                //printf(",%e", hidden_layer[i].weight[j]);
            }
        }

        fprintf(model_file, "\n");
        //printf("\n");

    }

    printf("Done!\n");
    printf("Saving output layer...\n");

    //save the output layer of the neural Network
    for(int i = 0; i < output_node_num; i++)
    {
        //record basic structure of the neuron
        fprintf(model_file, "%f,%f,%f", output_layer[i].states, output_layer[i].output, output_layer[i].error);
        //printf("%e,%e,%e", output_layer[i].states, output_layer[i].output, output_layer[i].error);

        //record the weight of the neuron
        for(int j = 0; j < hidden_node_num; j++)
        {
            fprintf(model_file, ",%f", output_layer[i].weight[j]);
            //printf(",%e", output_layer[i].weight[j]);
        }

        fprintf(model_file, "\n");
        //printf("\n");
    }

    printf("Done!\n");

    fclose(model_file);
}


/**
 * Description:
 *  The Feedforward Backpropagation Neural Network whole process program
 * Parameter:
 *
 */
void ffbpn(
    int input_node_num, int hidden_node_num, int hidden_layer_num, int output_node_num,
    double learningRate, double bias,
    FILE *trainData, int dataLen,
    int epochTime
)
{
    // Determine the acitvation function
    // 1. sigmoid
    // 2. Lee_oscillator
    int activation_type = 2;
    
    if(activation_type=2){
        printf("CONN Model Load");
        Load_Oscillator();
        Load_Oscillator_Derivation();
    }
    
    
    //create the NN model by three kinds of layer
    printf("Creating Neural Network Model...");
    neuron input_layer[input_node_num];
    neuron hidden_layer[hidden_layer_num*hidden_node_num];
    neuron output_layer[output_node_num];
    printf("Done!\n");

    //initialize each layer
    printf("Initializing Neural Network Model...");
    initInputlayer(input_layer, input_node_num);
    initHiddenLayer(hidden_layer, input_node_num, hidden_layer_num, hidden_node_num);
    initOutputLayer(output_layer, output_node_num, hidden_node_num);
    printf("Done!\n");

    //load the trainData
    printf("Reading Data from csv file...");
    int index;
    double *train = loadData(trainData, input_node_num, output_node_num, dataLen, &index);
    printf("Done!\n");

    //devide the data to input and output
    printf("Dividing training data to input and output...");
    double *input = (double *)malloc(index*input_node_num * sizeof(double));
    double *output = (double *)malloc(index*output_node_num * sizeof(double));

    if(input == NULL || output == NULL)
    {
        printf("Out of memery while initialize input and output data\n");
        return;
    }

    //devide the input layer and output layer
    for(int i = 0; i < index; i++){
        for(int m = 0; m < input_node_num; m++)
        {
            input[i*input_node_num+m] = train[i*(input_node_num+output_node_num)+m];
        }

        for(int n = input_node_num; n < input_node_num+output_node_num; n++)
        {
            output[i*output_node_num+(n-input_node_num)] = train[i*(input_node_num+output_node_num)+n];
        }
    }

    //create predict value
    double *predict = (double *)malloc(input_node_num * sizeof(double));
    int predict_index = 0;
    for(int j = (index - 9) * 4; j < index * 4; j++)
    {
        predict[predict_index] = train[j];
        predict_index++;
    }

    free(train);
    printf("Done!\n");
    //printf("There are total %d datas\n", index);

    //start training
    printf("Start Neural Network Training...\n");
    double cost = 0.0;
    double current_error = 0.0;
    int training_num = 1;


    for(int epoch = 1; epoch <= epochTime; epoch++)
    {
        // if(epoch == 35){
        //     printf("Debug....");
        // }
        for(int times = 0; times < index; times++)
        {
            //training proces

            //load the data to input layer
            loadInputLayer(input_layer, input_node_num, input);
            FeedForward(input_layer, hidden_layer, output_layer, input_node_num, hidden_node_num, output_node_num, hidden_layer_num, bias, activation_type);
            cost = backpropagation(input_layer, hidden_layer, output_layer, output, input_node_num, hidden_node_num, output_node_num, hidden_layer_num, learningRate, bias, activation_type);
            current_error = cost / (training_num + 1);

            printf("\rEpoch times: %d, Training %d/%d data, MSE: %e", epoch, times+1, index, current_error);
            training_num++;
        }

        printf("\n");
    }

    printf("Training finish\n");

    //tracking the value of the hidden layer
    //checkHidden(hidden_layer, input_node_num, hidden_node_num, hidden_layer_num);

    //save the neural network model
    printf("Prepare to save the NN model\n");
    saveModel(input_layer, hidden_layer, output_layer, input_node_num, hidden_layer_num, hidden_node_num, output_node_num, learningRate, bias);

    //using the last record to predict new value
    printf("Using the newest data to predict tomorrow value:\n");
    loadInputLayer(input_layer, input_node_num, predict);
    FeedForward(input_layer, hidden_layer, output_layer, input_node_num, hidden_node_num, output_node_num, hidden_layer_num, bias, activation_type);
    for(int q = 0; q < output_node_num; q++)
    {
        printf("\t%f", output_layer[q].output);
    }
    printf("\n");
}


/**
 * Description:
 * 
 */
nn * loadModel()
{
    nn savingModel;

    FILE *model_file = fopen("ffbpn.model", "r");



    return NULL;
}


// //for test the function
// int main(){
//     srand(time(NULL));
//     for(int i = 0; i < 5; i++)
//     {
//         printf("\tnumber = %f\n", random_0_1());
//     }
// }