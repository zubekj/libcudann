/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

/*
 * ActivationFunctions.h
 *
 *  Created on: Nov 24, 2010
 *      Author: donati
 */

#ifndef ACTIVATIONFUNCTIONS_H_
#define ACTIVATIONFUNCTIONS_H_

#define ACT_LINEAR		0
#define ACT_SIGMOID		1
#define ACT_TANH		2

//returns the value of the activation function
float actFunction(int act, float x);
//returns the value of the activation function derivation (used for backpropagation)
float actDerivation(int act, float y);
//returns the span size of a function for error calculation
float spanSize(int act);

/*//old macros
#define clip(x, lo, hi) (((x) < (lo)) ? (lo) : (((x) > (hi)) ? (hi) : (x)))

#define spanSize(act)(\
	act == ACT_TANH ? 2:\
	1\
)

#define actFunction(act, x)(\
	act == ACT_LINEAR ?		x															:\
	act == ACT_SIGMOID ?	1.0f/(1.0f+exp(-x))											:\
	act == ACT_TANH ?		2.0f/(1.0f+exp(-x))-1.0f									:\
	0\
)

#define actDerivation(act, y)(\
	act == ACT_LINEAR ?		1															:\
	act == ACT_SIGMOID ?	clip(y,0.01f,0.99f)*(1.0f-clip(y,0.01f,0.99f))				:\
	act == ACT_TANH ?		0.5f*(1.0f-(clip(y,-0.98f,0.98f)*clip(y,-0.98f,0.98f)))		:\
	0\
)
*/

#endif /* ACTIVATIONFUNCTIONS_H_ */
