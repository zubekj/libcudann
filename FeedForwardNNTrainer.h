/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

/*
 * FeedForwardNNTrainer.h
 *
 *  Created on: 19/nov/2010
 *      Author: donati
 */

#ifndef FEEDFORWARDNNTRAINER_H_
#define FEEDFORWARDNNTRAINER_H_

#include "FeedForwardNN.h"
#include "LearningSet.h"
#include "ErrorFunctions.h"
#include "CudaActivationFunctions.cuh"
#include "CudaErrorFunctions.cuh"

#define TRAIN_CPU 0
#define TRAIN_GPU 1
#define ALG_BP 0
#define ALG_BATCH 1
#define SHUFFLE_OFF 0
#define SHUFFLE_ON 1

#define PRINT_ALL 0
#define PRINT_MIN 1
#define PRINT_OFF 2

class FeedForwardNNTrainer {
public:
	FeedForwardNNTrainer();
	virtual ~FeedForwardNNTrainer();
	//choose a net to operate on and save after the training
	void selectNet(FeedForwardNN &);
	//choose the training set
	void selectTrainingSet(LearningSet &);
	//choose the test set. if this is set the error rate is computed on test set instead of training set
	void selectTestSet(LearningSet &);
	//choose a net to save the best network trained so far after each epoch. mse on test set is the criterion
	void selectBestMSETestNet(FeedForwardNN &);
	//choose a net to save the best network trained so far after each epoch. mse on train set + mse on test set is the criterion
	void selectBestMSETrainTestNet(FeedForwardNN &);
	//choose a net to save the best network trained so far after each epoch. percentage as classifier is the criterion
	void selectBestClassTestNet(FeedForwardNN &);
	//starts the training using params. n is the number of parameters
	//the first 2 elements of params are where the training will be executed (TRAIN_CPU,TRAIN_GPU)
	//and the training algorithm (ALG_BP,ALG_BATCH...). the other parameters are algorithm dependent
	//returns the best MSE on test set (or train set if test set isn't specified)
	//printtype specifies how much verbose will be the execution (PRINT_ALL,PRINT_MIN,PRINT_OFF)
	float train(const int n, const float * params,const int printtype=PRINT_ALL);
private:
	//backpropagation training on host
	//n is the number of parameters. parameters are (float array):
	//desired error, max_epochs, epochs_between_reports, learning_rate, momentum (using momentum is 20% slower), shuffle (SHUFFLE_ON or SHUFFLE_OFF), error function (ERROR_TANH or ERROR_LINEAR)
	float trainCpuBp(const int n, const float * params, const int printtype);
	//batch training on host
	//n is the number of parameters. parameters are (float array):
	//desired error, max_epochs, epochs_between_reports, learning_rate, momentum (using momentum is 20% slower), shuffle (SHUFFLE_ON or SHUFFLE_OFF), error function (ERROR_TANH or ERROR_LINEAR)void FeedForwardNNTrainer::trainCpuBatch(const int n, const float * params){
	float trainCpuBatch(const int n, const float * params, const int printtype);
	//batch training on device
	//n is the number of parameters. parameters are (float array):
	//desired error, max_epochs, epochs_between_reports, learning_rate, momentum (using momentum is 20% slower), shuffle (SHUFFLE_ON or SHUFFLE_OFF), error function (ERROR_TANH or ERROR_LINEAR)void FeedForwardNNTrainer::trainCpuBatch(const int n, const float * params){
	float trainGPUBatch(const int n, const float * params, const int printtype);
	//computes a single instance forward of the backpropagation training
	void stepForward(float * values, const  float * weights, const  int * actFuncts, const  int numOfLayers, const  int * layersSize, const  int numOfInputsPerInstance, const float * trainingSetInputs, const int * offsetIns, const int * offsetWeights, const int * offsetOuts, const int * order, const int instance);
	//computes a single instance backward of the backpropagation training
	void stepBack(const float * values, const  float * weights, float * deltas,  const  int * actFuncts, const  int numOfLayers, const  int * layersSize, const  int numOfOutputsPerInstance, const float * trainingSetOutputs, const int * offsetWeights, const int * offsetDeltas, const int * offsetOuts, const int * order, const int instance, const int errorFunc);
	//update the weights using the deltas
	void weightsUpdate(const float * values, const float * weights, float * weightsToUpdate, const float * deltas, const  int numOfLayers, const  int * layersSize, const int * offsetIns, const int * offsetWeights, const int * offsetDeltas, const float momentum, float * oldWeights, float learningRate);
	//GPU computes all the instances forward of the backpropagation training
	void GPUForward(float * devValues, const  float * devWeights, const  int * actFuncts, const  int numOfLayers, const  int * layersSize, const int numOfInstances, const int * offsetIns, const int * offsetWeights, const int * offsetOuts);
	//GPU computes all the instances backward of the backpropagation training
	void GPUBack(const float * devValues,const float * devWeights,float * devDeltas,const int * actFuncts,const int numOfLayers,const int *layersSize,const int numOfInstances,const int numOfOutputsPerInstance,const float * devTrainingSetOutputs,const int *offsetWeights,const int *offsetDeltas,const int * offsetOuts, const int errorFunc);
	//GPU updates the weights for all the instances
	void GPUUpdate(const float * devValues,float * devWeights,const float *devDeltas, const int numOfLayers, const int * layersSize, const int numOfInstances, const int * offsetIns,const int * offsetWeights,const int * offsetDeltas,const float momentum,float * devOldWeights,const float learningRate);
	//GPU computes the MSE on a set
	float GPUComputeMSE(float * devValues, const  float * devWeights, const  int * actFuncts, const  int numOfLayers, const  int * layersSize, const int numOfInstances, const int numOfOutputsPerInstance,const float * devSetOutputs,const int * offsetIns, const int * offsetWeights, const int * offsetOuts);
	//GPU computes the classification percentage on a set
	float GPUclassificatePerc(float * devValues, const  float * devWeights, const  int * actFuncts, const  int numOfLayers, const  int * layersSize, const int numOfInstances, const int numOfOutputsPerInstance,float * devSetOutputs,const int * offsetIns, const int * offsetWeights, const int * offsetOuts);

	FeedForwardNN * net;
	LearningSet * trainingSet;
	LearningSet * testSet;
	FeedForwardNN * bestMSETestNet;
	FeedForwardNN * bestMSETrainTestNet;
	FeedForwardNN * bestClassTestNet;
};

#endif /* FEEDFORWARDNNTRAINER_H_ */
