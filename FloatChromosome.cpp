/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

/*
 * FloatChromosome.cpp
 *
 *  Created on: Jan 20, 2011
 *      Author: donati
 */

#include "FloatChromosome.h"

#define NULL 0

#include <stdlib.h>
#include <sys/time.h>


FloatChromosome::FloatChromosome() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	srand(tv.tv_usec);

	size=0;
	values=NULL;
}

FloatChromosome::~FloatChromosome() {
	delete []values;
}

FloatChromosome::FloatChromosome(const int n) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	srand(tv.tv_usec);

	size=n;
	values=new float[n];
}


// copy constructor
FloatChromosome::FloatChromosome(const FloatChromosome & oldChr){
	size=oldChr.size;
	values = new float[size];
	for(int i=0;i<size;i++){
		values[i]=oldChr.values[i];
	}
}
// assignment operator
FloatChromosome & FloatChromosome::operator = (const FloatChromosome & oldChr){
	if (this != &oldChr){ // protect against invalid self-assignment
		// 1: allocate new memory and copy the elements
		float * new_values = new float[oldChr.size];
		for(int i=0;i<oldChr.size;i++){
			new_values[i]=oldChr.values[i];
		}
		// 2: deallocate old memory
		delete [] values;
		// 3: assign the new memory to the object
		values=new_values;
		size=oldChr.size;
		}
		// by convention, always return *this
	return *this;
}


int FloatChromosome::getSize() {
	return size;
}
void FloatChromosome::setSize(const int n) {
	size=n;
	delete [] values;
	values=new float[n];
}
float FloatChromosome::getElement(const int i){
	return values[i];
}
void FloatChromosome::setElement(const int i, const float el){
	values[i]=el;
}


