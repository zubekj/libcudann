/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

/*
 * ActivationFunctions.cpp
 *
 *  Created on: Nov 24, 2010
 *      Author: donati
 */

#include "ActivationFunctions.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define clip(x, lo, hi) (((x) < (lo)) ? (lo) : (((x) > (hi)) ? (hi) : (x)))

//returns the value of the activation function
float actFunction(int act, float x){
	switch(act){
		case ACT_LINEAR:	return x;//printf("LINEAR SHOULD NOT BE USED FOR NOW\n");exit(1);
		case ACT_SIGMOID:	return 1.0f/(1.0f+exp(-x));
		case ACT_TANH:		return 2.0f/(1.0f+exp(-x))-1.0f;
		default:			printf("FUNCTION NOT IMPLEMENTED YET\n");exit(1);
	}
}
//returns the value of the activation function derivation (used for backpropagation)
float actDerivation(int act, float y){
	switch(act){
		case ACT_LINEAR:	return 1;//printf("LINEAR SHOULD NOT BE USED FOR DERIVATION\n");exit(1);
		case ACT_SIGMOID:	y=clip(y,0.01f,0.99f);return y*(1.0f-y);
		case ACT_TANH:		y=clip(y,-0.98f,0.98f);return 0.5f*(1.0f-(y*y));
		default:			printf("FUNCTION NOT IMPLEMENTED YET\n");exit(1);
	}
}
//returns the span size of a function for error calculation
float spanSize(int act){
	switch(act){
		case ACT_LINEAR:
		case ACT_SIGMOID:	return 1.0f;
		case ACT_TANH:		return 2.0f;
		default:			printf("FUNCTION NOT IMPLEMENTED YET\n");exit(1);
	}
}

