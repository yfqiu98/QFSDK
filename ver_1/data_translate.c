#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int array[3] = {1, 2, 3};

void modify(){
    for(int i = 0; i < 3; i++){
        array[i]++;
    }
}

int main(){
    FILE *fp = fopen("CADCHF1440.csv", "r");
    FILE *fw = fopen("Data_4_mix_1.csv", "a");

    char line[48];
    char *result;

    int time = 0;
    double input[36];
    double output[4];

    while(fgets(line, 48, fp)){
        double current_data[40];
        result = strtok(line, ",");
        int k = 0;
        while(result != NULL){
            current_data[k] = atof(result);
            result = strtok(NULL, ",");
            //printf("\t%f", current_data[k]);
            k++;
        }

        if(time == 9){
            time = 0;
            for(int i = 0; i < 4; i++){
                output[i] = current_data[i+3];
                fprintf(fw, "%f,", output[i]);
                printf("\t%f", output[i]);
            }
            fprintf(fw, "\n");
            printf("\n");
        }
        else
        {
            for(int i = 0; i < 4; i++){
                input[i + time] = current_data[i+3];
                fprintf(fw, "%f,", input[i + time]);
                printf("\t%f", input[i + time]);
            }
            time++;
        }
    }
    

    return 0;
}