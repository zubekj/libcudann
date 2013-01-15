/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

#include <stdio.h>
#include <iostream>

#include <math.h>
#include <stdlib.h>

#include <libcudann/libcudann.h>

using namespace std;

int main(){

	//TRAINING EXAMPLE

	LearningSet trainingSet("mushroom.train");
	LearningSet testSet("mushroom.test");

	//layer sizes
	int layers[]={125,30,2};
	//activation functions (1=sigm,2=tanh)
	int functs[]={1,1,1};
	//declare the network with the number of layers
	FeedForwardNN mynet(3,layers,functs);

        mynet.initWeights();

	FeedForwardNNTrainer trainer;
	trainer.selectNet(mynet);
	trainer.selectTrainingSet(trainingSet);
	trainer.selectTestSet(testSet);

	//optionally save best net found on test set, or on train+test, or best classifier
	//FeedForwardNN mseT;
	//FeedForwardNN mseTT;
	//FeedForwardNN cl;
	//trainer.selectBestMSETestNet(mseT);
	//trainer.selectBestMSETrainTestNet(mseTT);
	//trainer.selectBestClassTestNet(cl);

	//parameters:
	//TRAIN_GPU - TRAIN_CPU
	//ALG_BATCH - ALG_BP (batch packpropagation or standard)
	//desired error
	//total epochs
	//epochs between reports
	//learning rate
	//momentum
	//SHUFFLE_ON - SHUFFLE_OFF
	//error computation ERROR_LINEAR - ERROR_TANH
	
	float param[]={TRAIN_GPU,ALG_BATCH,0.001,2000,500,0.7,0,SHUFFLE_ON,ERROR_TANH};
	trainer.train(9,param);
	 
        printf("Accuracy: %f\n", mynet.classificatePerc(testSet));
	
	mynet.saveToTxt("mynetmushrooms.net");


	//mseT.saveToTxt("../mseTmushrooms.net");
	//mseTT.saveToTxt("../mseTTmushrooms.net");
	//cl.saveToTxt("../clmushrooms.net");


/*	//EVOLUTION EXAMPLE

	LearningSet trainingSet("mushroom.train");
	LearningSet testSet("mushroom.test");

	GAFeedForwardNN evo;

	//choose a net to save the best training
	FeedForwardNN mybest;
	evo.selectBestNet(mybest);

	evo.selectTrainingSet(trainingSet);
	evo.selectTestSet(testSet);

	//evolution parameters:
	//popolation
	//generations
	//selection algorithm ROULETTE_WHEEL - TOURNAMENT_SELECTION
	//training for each generated network
	//crossover probability
	//mutation probability
	//number of layers
	//max layer size
	evo.init(5,10,ROULETTE_WHEEL,2,0.5,0.3,1,50);

	//training parameters:
	//TRAIN_GPU - TRAIN_CPU
	//ALG_BATCH - ALG_BP (batch packpropagation or standard)
	//desired error
	//total epochs
	//epochs between reports

	float param[]={TRAIN_GPU,ALG_BATCH,0.00,300,100};

	evo.evolve(5,param,PRINT_MIN);

	mybest.saveToTxt("mybestmushroom.net"); /*

/*	//USAGE EXAMPLE
	//load a trained network from a file
	FeedForwardNN net("mytrainedxornet.net");
	float in[2],out[1];
	in[0]=1;
	in[1]=0;
	
	//compute the network (for example an xor function) from inputs in[0] and in[1] and puts the result in out[0]
	net.compute(in,out);
	printf("%f\n",out[0]);
*/
}






