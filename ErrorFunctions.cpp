/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

/*
 * ErrorFunctions.cpp
 *
 *  Created on: Dec 22, 2010
 *      Author: donati
 */

#include "ErrorFunctions.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

//returns the new error after the application of a function (tanh is more aggressive error targeting)
float errorFunction(float error, int func){
	switch(func){
			case ERROR_TANH:	if		(error < -.9999999)			return -17.0;
								else if	(error >  .9999999)			return 17.0;
								else 								return log((1.0 + error) / (1.0 - error));
			case ERROR_LINEAR:	return error;
			default:			printf("FUNCTION NOT IMPLEMENTED YET\n");exit(1);
		}
}
