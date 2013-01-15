/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

/*
 * LearningSet.h
 *
 *  Created on: 19/nov/2010
 *      Author: donati
 */

#ifndef LEARNINGSET_H_
#define LEARNINGSET_H_

class LearningSet {
public:
	LearningSet();
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
	LearningSet(const char *);
	// copy constructor
	LearningSet(const LearningSet &);
        // create from array data
        LearningSet(int, int, int, const float*, const float*);
   	// assignment operator
	LearningSet & operator = (const LearningSet &);
	virtual ~LearningSet();
    float *getInputs() const;
    int getNumOfInputsPerInstance() const;
    int getNumOfInstances() const;
    int getNumOfOutputsPerInstance() const;
    float *getOutputs() const;
private:
	int numOfInstances;
	int numOfInputsPerInstance;
	int numOfOutputsPerInstance;
	float * inputs;
	float * outputs;
};

#endif /* LEARNINGSET_H_ */
