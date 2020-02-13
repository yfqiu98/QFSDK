#include "ffbpn.h"

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

int main()
{
    FILE *trade = fopen("USDCAD-1.csv", "r");
    FILE *qpl = fopen("qpl.csv", "r");

    double *tradeData = (double *)malloc(2049*4*sizeof(double));

    char *line = (char *)malloc(4 * 10 * sizeof(10));
    char *result;

    int index = 0;

    while (fgets(line, 40, trade) != NULL)
    {
        result = strtok(line, ",");
        int k = 0;
        while (result != NULL)
        {
            tradeData[index * 4 + k] = atof(result);
            result = strtok(NULL, ",");
            k++;
        }
        index++;
    }
    
    double *qpl = (double *)malloc(21*sizeof(double));

}