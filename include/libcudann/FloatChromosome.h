/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

/*
 * FloatChromosome.h
 *
 *  Created on: Jan 20, 2011
 *      Author: donati
 */

#ifndef FLOATCHROMOSOME_H_
#define FLOATCHROMOSOME_H_

class FloatChromosome {
public:
	FloatChromosome();
	virtual ~FloatChromosome();
	FloatChromosome(const int n);
	// copy constructor
	FloatChromosome(const FloatChromosome &);
	//assignment operator
	FloatChromosome & operator = (const FloatChromosome &);

	void setSize(const int n);
	int getSize();
	float getElement(const int i);
	void setElement(const int i, const float el);

private:
	int size;
	float * values;
};

#endif /* FLOATCHROMOSOME_H_ */
