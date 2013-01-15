/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

/*
 * CudaErrorFunctions.cu
 *
 *  Created on: Jan 10, 2011
 *      Author: donati
 */

#include "CudaErrorFunctions.cuh"

#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>


#define BLOCKSIZE 512
#define WARP_SIZE 32



__global__ void error(float * c, const float * a, const float * b, const int number, const int actFunc, const int errorFunc){
	//global thread index
	const int g_tid = BLOCKSIZE * blockIdx.x + threadIdx.x;
	if(g_tid<number){
		float error=(a[g_tid]-b[g_tid])/spanS(actFunc);
		c[g_tid]=calcErr(error,errorFunc);
	}
}
//computes the error function for (number) elements of (a)-(b) and store the results in (c)
void computeError(float * c, const float * a, const float * b, const int number, const int actFunc, const int errorFunc){

int numBlocks = number/BLOCKSIZE+1;
error<<<numBlocks, BLOCKSIZE>>>(c,a,b,number,actFunc,errorFunc);

}


//computes the total mse for (number) elements of (desired)-(neurons)
float mseError(const float * desired, float * neurons, const int number, const int actFunc){

int numBlocks = number/BLOCKSIZE+1;
error<<<numBlocks, BLOCKSIZE>>>(neurons,desired,neurons,number,actFunc,ERROR_LINEAR);
//does the product of each member then sums them all and divides for number
return cublasSdot(number, neurons, 1, neurons, 1)/(float)number;
}


__global__ void maxes(const int nOfInst, const int nOfOut, const float * neurons, int * indexes){
	extern __shared__ float sdata[];
	float *sidx = sdata + BLOCKSIZE;

	const int tid		= threadIdx.x;					// thread index
	const int thread_lane	= tid & (WARP_SIZE-1);		// thread index within the warp
	const int g_tid		= BLOCKSIZE * blockIdx.x + tid;	// global thread index
	const int g_warp_id	= g_tid / WARP_SIZE;			// global warp index

	const int offset=g_warp_id*nOfOut;

	//loading in shared data of values
	sdata[tid]=((offset+thread_lane<nOfInst*nOfOut)&&(thread_lane<nOfOut))?neurons[offset+thread_lane]:0.0f;
	//loading in shared data of indexes
	sidx[tid]=thread_lane;

	if(g_warp_id<nOfInst){

		//some sequential reduction (suggested to maximize the throughput)
		for(unsigned int i = thread_lane+WARP_SIZE ; i < nOfOut ; i += WARP_SIZE){
			float aux=neurons[offset+i];
			if(sdata[tid] < aux){sdata[tid]=aux;sidx[tid]=i;}
		}

		//parallel reduction of both the value and the index
		if (thread_lane < 16){
			if(sdata[tid] < sdata[tid+16]){sdata[tid]=sdata[tid+16];sidx[tid]=sidx[tid+16];}
			if(sdata[tid] < sdata[tid+8]){sdata[tid]=sdata[tid+8];sidx[tid]=sidx[tid+8];}
			if(sdata[tid] < sdata[tid+4]){sdata[tid]=sdata[tid+4];sidx[tid]=sidx[tid+4];}
			if(sdata[tid] < sdata[tid+2]){sdata[tid]=sdata[tid+2];sidx[tid]=sidx[tid+2];}
			if(sdata[tid] < sdata[tid+1]){sdata[tid]=sdata[tid+1];sidx[tid]=sidx[tid+1];}
		}

		//return the best neuron index
		if (thread_lane == 0){
			indexes[g_warp_id]=sidx[tid];
		}

	}
}
//find the (indexes) of the max values of each row of a set of (neurons), divided in rows(nOfOut) and columns(nOfInst)
void computeMaxes(const int nOfInst, const int nOfOut, const float * neurons, int * indexes){

int numBlocks = nOfInst / (BLOCKSIZE/WARP_SIZE)+1;
int smemSize = 2 * BLOCKSIZE  * sizeof(float);

maxes<<<numBlocks, BLOCKSIZE,smemSize>>>(nOfInst, nOfOut, neurons, indexes);

}



__global__ void addMom(float * weights, float * oldWeights,const int number, const float momentum){
	//global thread index
	const int g_tid = BLOCKSIZE * blockIdx.x + threadIdx.x;
	const float weight=weights[g_tid];
	if(g_tid<number){
		weights[g_tid]+=momentum*(weight-oldWeights[g_tid]);
		oldWeights[g_tid]=weight;
	}
}
//adds to (number) elements of (weights) the difference between (weights) and (oldWeights) multiplied with (momentum). also update (oldWeights)
void addMomentum(float * weights, float * oldWeights,const int number, const float momentum){

int numBlocks = number/BLOCKSIZE+1;

addMom<<<numBlocks, BLOCKSIZE>>>(weights,oldWeights,number,momentum);

}


__global__ void trMatrix(const int x, const int y, const float * in, float * out){
	//global thread index
	const int g_tid = BLOCKSIZE * blockIdx.x + threadIdx.x;

	if(g_tid<x*y){
		out[g_tid%x*y+g_tid/x]=in[g_tid];
	}
}
//translate a matrix x-y (rows large (x) and columns high (y)) to one y-x
void translateMatrix(const int x, const int y, const float * in, float * out){

int numBlocks = (x*y)/BLOCKSIZE+1;

trMatrix<<<numBlocks, BLOCKSIZE>>>(x,y,in,out);

}

