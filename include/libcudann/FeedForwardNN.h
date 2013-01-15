/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

/*
 * FeedForwardNN.h
 *
 *  Created on: Nov 17, 2010
 *      Author: donati
 */

#ifndef FEEDFORWARDNN_H_
#define FEEDFORWARDNN_H_

#include "ActivationFunctions.h"
#include "LearningSet.h"

#define INITWEIGHTMAX 0.1

class FeedForwardNN {
public:
	FeedForwardNN();
	// constructor with int (number of layers), array (layer sizes), array (activation functions)
	FeedForwardNN(const int, const int *, const int *);
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
	FeedForwardNN(const char *);
	// copy constructor
	FeedForwardNN(const FeedForwardNN &);
	//assignment operator
	FeedForwardNN & operator = (const FeedForwardNN &);
	virtual ~FeedForwardNN();
	// initialize randomly the network weights between min and max
	void initWeights(float min =-INITWEIGHTMAX,float max = INITWEIGHTMAX);
	// initialize the network weights with Widrow Nguyen algorithm
	void initWidrowNguyen(LearningSet &);
	// computes the net outputs
	void compute(const float *, float *);
	// computes the MSE on a set
	float computeMSE(LearningSet &);
	// returns the index of the most high output neuron (classification)
	int classificate(const float * inputs);
	// computes the correct percentage of classification on a set (0 to 1)
	float classificatePerc(LearningSet &);
	// saves the network to a txt file
	void saveToTxt(const char *);
	float getWeight(int ind) const;
    void setWeight(int ind, float weight);
	int *getLayersSize() const;
    int getNumOfLayers() const;
    int getNumOfWeights() const;
    float *getWeights() const;
    int *getActFuncts() const;
private:
	int numOfLayers;
	int * layersSize;
	int * actFuncts;
	int numOfWeights;
	float * weights;
};

#endif /* FEEDFORWARDNN_H_ */
