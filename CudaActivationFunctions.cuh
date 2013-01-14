/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

/*
 * CudaActivationFunctions.cuh
 *
 *  Created on: Jan 10, 2011
 *      Author: donati
 */

#ifndef CUDAACTIVATIONFUNCTIONS_H_
#define CUDAACTIVATIONFUNCTIONS_H_

#define ACT_LINEAR		0
#define ACT_SIGMOID		1
#define ACT_TANH		2

//macro for the span size of the function (for error calculation of backpropagation)
#define spanS(act)(\
	act == ACT_TANH ? 2:\
	1\
)

//computes the activation function for (number) elements of (neurons) and store the results in (neurons)
void computeActFunct(float * neurons, const int number, const int funct);

//computes the derivation function for (number) elements of (neurons) and multiplies and stores the results with and in (delta)
void computeDerivFunct(float * deltas, const float * neurons, const int number, const int funct);

#endif /* CUDAACTIVATIONFUNCTIONS_H_ */
