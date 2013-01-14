/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

/*
 * GAFeedForwardNN.h
 *
 *  Created on: Jan 20, 2011
 *      Author: donati
 */

#ifndef GAFEEDFORWARDNN_H_
#define GAFEEDFORWARDNN_H_

#include "FeedForwardNN.h"
#include "LearningSet.h"
#include "FloatChromosome.h"
#include "FeedForwardNNTrainer.h"


#define ROULETTE_WHEEL 0
#define TOURNAMENT_SELECTION 1

//performs crossover between a chromosome and a mate, and return the result
FloatChromosome crossover(const FloatChromosome & first, const FloatChromosome & second, const float pcross);
//performs mutation on a chromosome, and return the result
FloatChromosome mutation(const FloatChromosome & first, const float pmut, const int maxdimhiddenlayers, const int nhidlayers);

//roulette wheel selection
int rouletteWheel(const int size, const float * fitnesses);

//tournament selection
int tournament(const int size, const float * fitnesses, const int toursize);

class GAFeedForwardNN {
public:
	GAFeedForwardNN();
	virtual ~GAFeedForwardNN();
	//choose the training set
	void selectTrainingSet(LearningSet &);
	//choose the test set. if this is set the error rate is computed on test set instead of training set
	void selectTestSet(LearningSet &);
	//choose a net to save the best network individual trained so far. mse on test set (or train if no test is specified) is the criterion
	void selectBestNet(FeedForwardNN & n);
	//initialize the genetic algorithm parameters and create the first population of individuals
	void init(const int popsize,const int generations,const int selectionalgorithm,const int numberofevaluations,const float pcross,const float pmut,const int nhidlayers,const int maxdimhiddenlayers);
	//run the genetic algorithm initialized before with some training parameters:
	//training location, training algorithm, desired error, max_epochs, epochs_between_reports
	//see "FeedForwardNNTrainer" class for more details
	//printtype specifies how much verbose will be the execution (PRINT_ALL,PRINT_MIN,PRINT_OFF)
	void evolve(const int n, const float * params, const int printtype=PRINT_ALL);

private:
	FeedForwardNN * net;
	LearningSet * trainingSet;
	LearningSet * testSet;
	FeedForwardNN * bestNet;

	FloatChromosome * chromosomes;

	int popsize;
	int generations;
	int numberofevaluations;
	int selectionalgorithm;
	float pcross;
	float pmut;
	int nhidlayers;
	int maxdimhiddenlayers;
};

#endif /* GAFEEDFORWARDNN_H_ */
