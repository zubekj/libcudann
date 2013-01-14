/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

/*
 * ErrorFunctions.h
 *
 *  Created on: Dec 22, 2010
 *      Author: donati
 */

#ifndef ERRORFUNCTIONS_H_
#define ERRORFUNCTIONS_H_

#define ERROR_LINEAR 0
#define ERROR_TANH 1

//returns the new error after the application of a function (tanh is more aggressive error targeting)
float errorFunction(float, int);

#endif /* ERRORFUNCTIONS_H_ */
