cdef extern from "libcudann/libcudann.h":

   cdef int ALG_BP, ALG_BATCH, SHUFFLE_ON, SHUFFLE_OFF, ERROR_TANH, ERROR_LINEAR
   cdef int TRAIN_CPU, TRAIN_GPU, PRINT_ALL, PRINT_MIN, PRINT_OFF

   cdef cppclass FeedForwardNN:
      FeedForwardNN()
      FeedForwardNN(int, int*, int*) except +
      FeedForwardNN(char*) except +
      void initWeights(float, float)
      void initWeights()
      void initWidrowNguyen(LearningSet)
      void compute(float*, float*)
      float computeMSE(LearningSet)
      int classificate(float*)
      float classificatePerc(LearningSet)
      void saveToTxt(char*)
      float getWeight(int)
      void setWeight(int, float)
      int *getLayersSize()
      int getNumOfLayers()
      int getNumOfWeights()
      float *getWeights()
      int *getActFuncts()
  
   cdef cppclass FeedForwardNNTrainer:
      FeedForwardNNTrainer()
      void selectNet(FeedForwardNN)
      void selectTrainingSet(LearningSet)
      void selectTestSet(LearningSet)
      void selectBestMSETestNet(FeedForwardNN)
      void selectBestMSETrainTestNet(FeedForwardNN)
      void selectBestClassTestNet(FeedForwardNN)
      float train(int, float*, int)
   
   cdef cppclass LearningSet:
      LearningSet()
      LearningSet(char*) except +
      LearningSet(LearningSet)
      LearningSet(int,int,int,float*,float*)
      float *getInputs()
      int getNumOfInputsPerInstance()
      int getNumOfInstances()
      int getNumOfOutputsPerInstance()
      float *getOutputs()
