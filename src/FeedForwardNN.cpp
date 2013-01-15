/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

/*
 * FeedForwardNN.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: donati
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "FeedForwardNN.h"

FeedForwardNN::FeedForwardNN(){
	numOfLayers=0;
	numOfWeights=0;
	layersSize=NULL;
	actFuncts=NULL;
	weights=NULL;
}
// constructor with int (number of layers), array (layer sizes), array (activation functions)
FeedForwardNN::FeedForwardNN(const int num, const int * siz, const int * funct) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	srand(tv.tv_usec);

	if(num<2){printf("BAD NETWORK INITIALIZATION\n");exit(1);}

	numOfLayers=num;
	layersSize = new int[numOfLayers];
	for(int i=0;i<numOfLayers;i++){
		layersSize[i]=siz[i];
	}
	actFuncts = new int[numOfLayers];
	for(int i=0;i<numOfLayers;i++){
		actFuncts[i]=funct[i];
	}
	numOfWeights = 0;
	for(int i=0;i<numOfLayers-1;i++){
		numOfWeights+=(layersSize[i]+1)*layersSize[i+1];
	}
	weights = new float[numOfWeights];
	initWeights();
}

/* constructor from txt file
 * format is:
 *
 * NUMBER_OF_LAYERS
 * LAYER1_SIZE LAYER2_SIZE LAYER3_SIZE ...
 * LAYER2_ACT_FUNC LAYER3_ACT_FUNC ...
 * NUMBER_OF_WEIGHTS
 * WEIGHT1
 * WEIGHT2
 * WEIGHT3
 * .
 * .
 * .
 *
 * spaces or \n do not matter
 */
FeedForwardNN::FeedForwardNN(const char * s){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	srand(tv.tv_usec);

	FILE * f;
	f=fopen(s,"r");

	//file not found
	if (f!=NULL){
		//file wrong format
		if(fscanf(f,"%d",&numOfLayers)<1){printf("WRONG NETWORK FILE FORMAT\n");exit(1);}
		layersSize = new int[numOfLayers];
		for(int i=0;i<numOfLayers;i++)
			if(fscanf(f,"%d",&layersSize[i])<1){printf("WRONG NETWORK FILE FORMAT\n");exit(1);}
		actFuncts = new int[numOfLayers];
		for(int i=0;i<numOfLayers;i++)
			if(fscanf(f,"%d",&actFuncts[i])<1){printf("WRONG NETWORK FILE FORMAT\n");exit(1);}
		if(fscanf(f,"%d",&numOfWeights)<1){printf("WRONG NETWORK FILE FORMAT\n");exit(1);}
		weights = new float[numOfWeights];
		for(int i=0;i<numOfWeights;i++)
			if(fscanf(f,"%f",&weights[i])<1){printf("WRONG NETWORK FILE FORMAT\n");exit(1);}
		fclose(f);
	}
	else{printf("COULDN'T OPEN THE NETWORK FILE\n");exit(1);}
}
// copy constructor
FeedForwardNN::FeedForwardNN(const FeedForwardNN & oldNet){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	srand(tv.tv_usec);

	numOfLayers=oldNet.numOfLayers;

	layersSize = new int[numOfLayers];
	for(int i=0;i<numOfLayers;i++){
		layersSize[i]=oldNet.layersSize[i];
	}

	actFuncts = new int[numOfLayers];
	for(int i=0;i<numOfLayers;i++){
		actFuncts[i]=oldNet.actFuncts[i];
	}
	numOfWeights = oldNet.numOfWeights;

	weights = new float[numOfWeights];
	for(int i=0;i<numOfWeights;i++){
		weights[i]=oldNet.weights[i];
	}
}
// assignment operator
FeedForwardNN & FeedForwardNN::operator = (const FeedForwardNN & oldNet){
	if (this != &oldNet){ // protect against invalid self-assignment
		// 1: allocate new memory and copy the elements

		int * new_layersSize = new int[oldNet.numOfLayers];
		for(int i=0;i<oldNet.numOfLayers;i++){
			new_layersSize[i]=oldNet.layersSize[i];
		}
		int * new_actFuncts = new int[oldNet.numOfLayers];
		for(int i=0;i<oldNet.numOfLayers;i++){
			new_actFuncts[i]=oldNet.actFuncts[i];
		}
		float * new_weights = new float[oldNet.numOfWeights];
		for(int i=0;i<oldNet.numOfWeights;i++){
			new_weights[i]=oldNet.weights[i];
		}
		// 2: deallocate old memory
		delete [] layersSize;
		delete [] actFuncts;
		delete [] weights;
		// 3: assign the new memory to the object
		layersSize=new_layersSize;
		actFuncts=new_actFuncts;
		weights=new_weights;
		numOfLayers=oldNet.numOfLayers;
		numOfWeights = oldNet.numOfWeights;
		}
		// by convention, always return *this
	return *this;
}

FeedForwardNN::~FeedForwardNN() {
	delete[] layersSize;
	delete[] actFuncts;
	delete[] weights;
}

// initialize randomly the network weights between min and max
void FeedForwardNN::initWeights(float min, float max)
{
	for(int i=0;i<numOfWeights;i++){
		//TEST WEIGHTS
		//weights[i]=2;
		//weights[i]=(2*max*((float)rand()/RAND_MAX))-max;
		weights[i]=(max-min)*((float)rand()/(RAND_MAX+1.0f))+min;

	}
}

// initialize the network weights with Widrow Nguyen algorithm
void FeedForwardNN::initWidrowNguyen(LearningSet & set){
	float min=set.getInputs()[0];
	float max=set.getInputs()[0];

	//finds the min and max value of inputs
	for (int i=0;i<set.getNumOfInstances()*set.getNumOfInputsPerInstance();i++){
		float val=set.getInputs()[i];
		if(val<min)
			min=val;
		if(val>max)
			max=val;
	}

	int nOfHid=0;
	for(int i=1;i<numOfLayers-1;i++)
		nOfHid+=layersSize[i];
	float mult=(float)(pow((double)(0.7f*(double)nOfHid),(double)(1.0f/(double)layersSize[0]))/(double)(max-min));


	int offsetWeights[numOfLayers];
	for(int i=0;i<numOfLayers;i++){
		offsetWeights[i] = 0;
		for(int j=0;j<i;j++){
			offsetWeights[i]+=(layersSize[j]+1)*layersSize[j+1];
		}
	}
	for(int i=0;i<numOfLayers-1;i++)
		for(int j=0;j<layersSize[i+1];j++)
			for(int k=0;k<layersSize[i]+1;k++)
				if(k<layersSize[i]){
					weights[offsetWeights[i]+j*(layersSize[i]+1)+k]=mult*((float)rand()/(RAND_MAX+1.0f));

				}
				else
					weights[offsetWeights[i]+j*(layersSize[i]+1)+k]=2*mult*((float)rand()/(RAND_MAX+1.0f))-mult;
}

// computes the net outputs
void FeedForwardNN::compute(const float * inputs, float * outputs){

	int offset = 0;
	float * in;
	float * out;

	//loads the inputs
	in = new float[layersSize[0]+1];
	for(int i=0;i<layersSize[0];i++)
		in[i]=inputs[i];

	out = new float[0];

	//loops the layers
	for(int i=0;i<numOfLayers-1;i++){

		//bias
		in[layersSize[i]]=1.0;

		offset=0;
		for(int j=0;j<i;j++){
			offset+=(layersSize[j]+1)*layersSize[j+1];
		}

		delete[] out;
		out = new float [layersSize[i+1]];

		float tot=0;


		//loops the outputs
		for(int j=0;j<layersSize[i+1];j++){
				tot=0;

				//loops the inputs
				for(int k=0;k<layersSize[i]+1;k++){
							tot+=in[k]*weights[k+j*(layersSize[i]+1)+offset];
						}
				out[j]=	actFunction(actFuncts[i+1],tot);
		}


		delete[] in;
		in = new float[layersSize[i+1]+1];
		for(int l=0;l<layersSize[i+1];l++){
				in[l]=out[l];
		}
	}
	delete[] in;

	for(int i=0;i<layersSize[numOfLayers-1];i++)
		outputs[i]=out[i];

	delete[] out;

}

// computes the MSE on a set
float FeedForwardNN::computeMSE(LearningSet & set){
	float mse=0;

	int numOfInstances=set.getNumOfInstances();
	int numOfInputsPerInstance=set.getNumOfInputsPerInstance();
	int numOfOutputsPerInstance=set.getNumOfOutputsPerInstance();

	float netOuts[numOfOutputsPerInstance];

	//local variables for faster access
	float * inputs=set.getInputs();
	float * outputs=set.getOutputs();

	for(int instance=0;instance<numOfInstances;instance++){
		//compute using the inputs with an offset to point to each instance
		compute(inputs+instance*numOfInputsPerInstance,netOuts);
		for(int i=0;i<numOfOutputsPerInstance;i++){
			float x=outputs[i+instance*numOfOutputsPerInstance]-netOuts[i];
			mse+=x*x;
		}
	}

	mse/=(numOfInstances*numOfOutputsPerInstance)*spanSize(actFuncts[numOfLayers-1])*spanSize(actFuncts[numOfLayers-1]);
	return mse;
}

// returns the index of the most high output neuron (classification)
int FeedForwardNN::classificate(const float * inputs){
	int outputsSize=layersSize[numOfLayers-1];
	float max=0;
	int indmax=0;
	float outputs[outputsSize];

	compute(inputs,outputs);
	for(int j=0;j<outputsSize;j++){
		if(outputs[j]>max){
			indmax=j;
			max=outputs[j];
		}
	}
	return indmax;
}

// computes the fraction of correct classification on a set (0 to 1)
float FeedForwardNN::classificatePerc(LearningSet & set){
	int cont=0;
	int numOfInstances=set.getNumOfInstances();
	int numOfInputsPerInstance=set.getNumOfInputsPerInstance();
	int numOfOutputsPerInstance=set.getNumOfOutputsPerInstance();

	for(int i=0;i<numOfInstances;i++){
		if(set.getOutputs()[classificate(set.getInputs()+i*numOfInputsPerInstance)+i*numOfOutputsPerInstance]==1){
			cont++;
		}
	}
	return (float)cont/(float)numOfInstances;
}

// saves the network to a txt file
void FeedForwardNN::saveToTxt(const char * s){
	FILE * f;
	f=fopen(s,"w");
	fprintf(f,"%d\n",numOfLayers);
	for(int i=0;i<numOfLayers;i++)
		fprintf(f,"%d ",layersSize[i]);
	fprintf(f,"\n");
	for(int i=0;i<numOfLayers;i++)
		fprintf(f,"%d ",actFuncts[i]);
	fprintf(f,"\n%d\n",numOfWeights);
	for(int i=0;i<numOfWeights;i++)
		fprintf(f,"%.20e\n",weights[i]);
	fclose(f);
}

int *FeedForwardNN::getLayersSize() const
{
    return layersSize;
}

int FeedForwardNN::getNumOfLayers() const
{
    return numOfLayers;
}

int FeedForwardNN::getNumOfWeights() const
{
    return numOfWeights;
}

float *FeedForwardNN::getWeights() const
{
    return weights;
}



int *FeedForwardNN::getActFuncts() const
{
    return actFuncts;
}

float FeedForwardNN::getWeight(int ind) const
{
	return weights[ind];
}

void FeedForwardNN::setWeight(int ind, float weight)
{
	weights[ind]=weight;
}



