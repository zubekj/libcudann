/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

/*
 * LearningSet.cpp
 *
 *  Created on: 19/nov/2010
 *      Author: donati
 */

#include <stdio.h>
#include <stdlib.h>

#include "LearningSet.h"


LearningSet::LearningSet(){
	numOfInstances=0;
	numOfInputsPerInstance=0;
	numOfOutputsPerInstance=0;
	inputs=NULL;
	outputs=NULL;
}
/* constructor from txt file (.fann format)
 * format is:
 *
 * NUMBER_OF_ISTANCES
 * NUMBER_OF_INPUTS_PER_ISTANCE
 * NUMBER_OF_OUTPUTS_PER_ISTANCE
 *
 * INPUT1 INPUT2 INPUT3 ...
 * OUTPUT1 OUTPUT2 OUTPUT3 ...
 *
 * INPUT1 INPUT2 INPUT3 ...
 * OUTPUT1 OUTPUT2 OUTPUT3 ...
 *
 * INPUT1 INPUT2 INPUT3 ...
 * OUTPUT1 OUTPUT2 OUTPUT3 ...
 *
 * .
 * .
 * .
 *
 * spaces or \n do not matter
 */
LearningSet::LearningSet(const char * s) {
	FILE * f;
		f=fopen(s,"r");
			//file not found
			if (f!=NULL){
				//file wrong format
				if(fscanf(f,"%d",&numOfInstances)<1){printf("WRONG LEARNING SET FILE FORMAT\n");exit(1);}
				if(fscanf(f,"%d",&numOfInputsPerInstance)<1){printf("WRONG LEARNING SET FILE FORMAT\n");exit(1);}
				if(fscanf(f,"%d",&numOfOutputsPerInstance)<1){printf("WRONG LEARNING SET FILE FORMAT\n");exit(1);}
				inputs = new float[numOfInstances*numOfInputsPerInstance];
				outputs = new float[numOfInstances*numOfOutputsPerInstance];
				for(int i=0;i<numOfInstances;i++){
					for(int j=0;j<numOfInputsPerInstance;j++)
						if(fscanf(f,"%f",&inputs[i*numOfInputsPerInstance+j])<1){printf("WRONG LEARNING SET FILE FORMAT\n");exit(1);}
					for(int j=0;j<numOfOutputsPerInstance;j++)
						if(fscanf(f,"%f",&outputs[i*numOfOutputsPerInstance+j])<1){printf("WRONG LEARNING SET FILE FORMAT\n");exit(1);}
				}
				fclose(f);
			}
			else{printf("COULDN'T OPEN THE LEARNING SET FILE\n");exit(1);}
}

// copy constructor
LearningSet::LearningSet(const LearningSet & oldSet){
	numOfInstances=oldSet.numOfInstances;
	numOfInputsPerInstance=oldSet.numOfInputsPerInstance;
	numOfOutputsPerInstance=oldSet.numOfOutputsPerInstance;
	inputs = new float[numOfInstances*numOfInputsPerInstance];
	outputs = new float[numOfInstances*numOfOutputsPerInstance];
	for(int i=0;i<numOfInstances;i++){
		for(int j=0;j<numOfInputsPerInstance;j++)
			inputs[i*numOfInputsPerInstance+j]=oldSet.inputs[i*numOfInputsPerInstance+j];
		for(int j=0;j<numOfOutputsPerInstance;j++)
			outputs[i*numOfOutputsPerInstance+j]=oldSet.outputs[i*numOfOutputsPerInstance+j];
	}
}

LearningSet & LearningSet::operator = (const LearningSet & oldSet){
	if (this != &oldSet){ // protect against invalid self-assignment
		// 1: allocate new memory and copy the elements
		numOfInstances=oldSet.numOfInstances;
		numOfInputsPerInstance=oldSet.numOfInputsPerInstance;
		numOfOutputsPerInstance=oldSet.numOfOutputsPerInstance;
		float * new_inputs = new float[numOfInstances*numOfInputsPerInstance];
		float * new_outputs = new float[numOfInstances*numOfOutputsPerInstance];
		for(int i=0;i<numOfInstances;i++){
			for(int j=0;j<numOfInputsPerInstance;j++)
				new_inputs[i*numOfInputsPerInstance+j]=oldSet.inputs[i*numOfInputsPerInstance+j];
			for(int j=0;j<numOfOutputsPerInstance;j++)
				new_outputs[i*numOfOutputsPerInstance+j]=oldSet.outputs[i*numOfOutputsPerInstance+j];
		}		// 2: deallocate old memory
		delete [] inputs;
		delete [] outputs;
		// 3: assign the new memory to the object
		inputs=new_inputs;
		outputs=new_outputs;
		}
		// by convention, always return *this
	return *this;
}

LearningSet::~LearningSet() {
	delete [] inputs;
	delete [] outputs;
}

float *LearningSet::getInputs() const
{
    return inputs;
}

int LearningSet::getNumOfInputsPerInstance() const
{
    return numOfInputsPerInstance;
}

int LearningSet::getNumOfInstances() const
{
    return numOfInstances;
}

int LearningSet::getNumOfOutputsPerInstance() const
{
    return numOfOutputsPerInstance;
}


float *LearningSet::getOutputs() const
{
    return outputs;
}
