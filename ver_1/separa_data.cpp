#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <windows.h>
int Decoder(FILE *File, int window_size, int feature_size, int dataLen, double *inputBox, double *outputBox);
int normalizer(double *input, double *output, int method);
int main(){

	int dataLen = 2048;
	int feature_size = 4;
	int window_size = 9;
	FILE *produce_USDCAD = fopen("USDCAD.csv", "r");

	double *input = (double *)malloc(dataLen * feature_size * window_size *sizeof(double));
	double *output = (double *)malloc(dataLen * feature_size *  sizeof(double));

	if(Decoder(produce_USDCAD, window_size, feature_size, dataLen, input, output)== 1){
		printf("load data to input and ouput array successfully ");
	}else{
		printf("Nope,nope,nope....");
	}
	fclose(produce_USDCAD);

	int d;
	scanf("%d",&d);
}
// this file is all about data preprocessing
// first this is used to transform CSV into different arrange. 
int Decoder(FILE *File, int window_size, int feature_size, int dataLen, double *inputBox, double *outputBox){

	if(File == NULL){
		return 0;
	}
	// to get all the data inside the File
	double *longList = (double *)malloc(2050 * 20 * sizeof(double));

	char temp[500];
	char *words;
	int daysize = 0;
	int attributes = 0;

	fgets(&temp[0],200,File);

	while(fgets(&temp[0],200,File)!= NULL){

		words = strtok(temp, ",");// seperate word by identifing ","

		while( words != NULL){

			longList[daysize * 20 + attributes] = strtod(words,NULL);
			words = strtok(NULL, ",");// seperate word by identifing ","
			attributes++;

		}

		attributes = 0;
		daysize++;
	}

	//sliding window to input and ouput
	int indexOfInput = 0;
	int indexOfOutput = 0;
	for(int x = 0;x < daysize-window_size-1;x++){
		//load to input array
		for(int y = 0;y < window_size;y++){
			//x
			for(int z = 0;z < feature_size;z++){
				inputBox[indexOfInput] = longList[(x+y)*20+z+3];
				indexOfInput++;
				
			}
		}
		// load to output array
		for(int z = 0;z < feature_size;z++){
			outputBox[indexOfOutput] = longList[(x+window_size)*20+z+3];
			indexOfOutput++;
		}
	}
	free(longList);
	return 1 ;

}
// and then, this is used to normalize
int normalizer(double *input, double *output, int method){
	return 0;
}