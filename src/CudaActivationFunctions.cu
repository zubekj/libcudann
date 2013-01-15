/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

/*
 * CudaActivationFunctions.cu
 *
 *  Created on: Jan 10, 2011
 *      Author: donati
 */

#include "CudaActivationFunctions.cuh"

#include <stdlib.h>
#include <stdio.h>

#define BLOCKSIZE 512

#define clip(x, lo, hi) (((x) < (lo)) ? (lo) : (((x) > (hi)) ? (hi) : (x)))

__global__ void actLinear(float * neurons, const int number){
	//global thread index
	const int g_tid = BLOCKSIZE * blockIdx.x + threadIdx.x;
	if(g_tid<number)
		neurons[g_tid]=neurons[g_tid];
}
__global__ void actSigmoid(float * neurons, const int number){
	//global thread index
	const int g_tid = BLOCKSIZE * blockIdx.x + threadIdx.x;
	if(g_tid<number)
		neurons[g_tid]=(1.0f/(1.0f+exp(-neurons[g_tid])));
}
__global__ void actTanh(float * neurons, const int number){
	//global thread index
	const int g_tid = BLOCKSIZE * blockIdx.x + threadIdx.x;
	if(g_tid<number)
		neurons[g_tid]=(2.0f/(1.0f+exp(-neurons[g_tid])))-1.0f;
}
__global__ void derivLinear(float * deltas, const float * neurons, const int number){
	//global thread index
	const int g_tid = BLOCKSIZE * blockIdx.x + threadIdx.x;
	if(g_tid<number){
		deltas[g_tid]*=1;
	}
}
__global__ void derivSigmoid(float * deltas, const float * neurons, const int number){
	//global thread index
	const int g_tid = BLOCKSIZE * blockIdx.x + threadIdx.x;
	if(g_tid<number){
		const float y=clip(neurons[g_tid],0.01f,0.99f);
		deltas[g_tid]*=y*(1.0f-y);
	}
}
__global__ void derivTanh(float * deltas, const float * neurons, const int number){
	//global thread index
	const int g_tid = BLOCKSIZE * blockIdx.x + threadIdx.x;
	if(g_tid<number){
		const float y=clip(neurons[g_tid],-0.98f,0.98f);
		deltas[g_tid]*=0.5f*(1.0f-(y*y));
	}
}

//computes the activation function for (number) elements of (neurons) and store the results in (neurons)
void computeActFunct(float * neurons, const int number, const int funct){

int numBlocks = number/BLOCKSIZE+1;

switch(funct){
	case ACT_LINEAR:	break;//actSigmoid<<<numBlocks, BLOCKSIZE>>>(neurons,number);//printf("LINEAR SHOULD NOT BE USED FOR NOW\n");exit(1);
	case ACT_SIGMOID:	actSigmoid<<<numBlocks, BLOCKSIZE>>>(neurons,number);	break;
	case ACT_TANH:		actTanh<<<numBlocks, BLOCKSIZE>>>(neurons,number);		break;
	default:			printf("FUNCTION NOT IMPLEMENTED YET\n");exit(1);			break;
}


}

//computes the derivation function for (number) elements of (neurons) and multiplies and stores the results with and in (delta)
void computeDerivFunct(float * deltas, const float * neurons, const int number, const int funct){

int numBlocks = number/BLOCKSIZE+1;

switch(funct){
	case ACT_LINEAR:	break;//derivSigmoid<<<numBlocks, BLOCKSIZE>>>(deltas,neurons,number);//printf("LINEAR SHOULD NOT BE USED FOR DERIVATION\n");exit(1);
	case ACT_SIGMOID:	derivSigmoid<<<numBlocks, BLOCKSIZE>>>(deltas,neurons,number);	break;
	case ACT_TANH:		derivTanh<<<numBlocks, BLOCKSIZE>>>(deltas,neurons,number);		break;
	default:			printf("FUNCTION NOT IMPLEMENTED YET\n");exit(1);			break;
}


}
