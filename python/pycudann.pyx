from libc.stdlib cimport malloc, free
cimport libcudann as cn

import numpy as np

SIGM = 1
TANH = 2
ERROR_TANH = cn.ERROR_TANH
ERROR_LINEAR = cn.ERROR_LINEAR
ALG_BP = cn.ALG_BP
ALG_BATCH = cn.ALG_BATCH
TRAIN_CPU = cn.TRAIN_CPU
TRAIN_GPU = cn.TRAIN_GPU
SHUFFLE_ON = cn.SHUFFLE_ON
SHUFFLE_OFF = cn.SHUFFLE_OFF
PRINT_ALL = cn.PRINT_ALL
PRINT_MIN = cn.PRINT_MIN
PRINT_OFF = cn.PRINT_OFF

def load_learning_set(str datafile):
   ls = LearningSet([[1]],[[1]])
   b_datafile = datafile.encode('UTF-8')
   cdef char *c_datafile = b_datafile
   ls.thisptr = new cn.LearningSet(c_datafile)
   return ls

cdef class LearningSet(object):

   cdef cn.LearningSet *thisptr

   def __cinit__(self, inputs, outputs):
      if len(inputs) != len(outputs):
         raise Exception()

      cdef int instances = len(inputs)
      cdef int input_size = len(inputs[0])
      cdef int output_size = len(outputs[0])

      cdef float *c_inputs = <float*>malloc(sizeof(float)*input_size*instances)
      for i in xrange(instances):
         if len(inputs[i]) != input_size:
            raise Exception()
         for j in xrange(input_size):
            c_inputs[i*input_size+j] = inputs[i][j]

      cdef float *c_outputs = <float*>malloc(sizeof(float)*output_size*instances)
      for i in xrange(instances):
         if len(outputs[i]) != output_size:
            raise Exception()
         for j in xrange(output_size):
            c_outputs[i*output_size+j] = outputs[i][j]

      self.thisptr = new cn.LearningSet(instances, input_size,
                           output_size, c_inputs, c_outputs)
      free(c_inputs)
      free(c_outputs)

   def __dealloc__(self):
      if self.thisptr is not NULL:
         del self.thisptr

   def getInputs(self):
      cdef int i,j
      cdef float *inputs = self.thisptr.getInputs()
      cdef int input_size = self.thisptr.getNumOfInputsPerInstance()
      cdef int instances = self.thisptr.getNumOfInstances()
      res = []
      for i in xrange(instances):
         row = [inputs[i*input_size + j] for j in xrange(input_size)]
         res.append(row)
      return res

   def getOutputs(self):
      cdef int i,j
      cdef float *outputs = self.thisptr.getOutputs()
      cdef int output_size = self.thisptr.getNumOfOutputsPerInstance()
      cdef int instances = self.thisptr.getNumOfInstances()
      res = []
      for i in xrange(instances):
         row = [outputs[i*output_size + j] for j in xrange(output_size)]
         res.append(row)
      return res

   def getNumOfInputsPerInstance(self):
      return self.thisptr.getNumOfInputsPerInstance()

   def getNumOfOutputsPerInstance(self):
      return self.thisptr.getNumOfOutputsPerInstance()

   def getNumOfInstances(self):
      return self.thisptr.getNumOfInstances()


def load_neural_net(str datafile):
   nn = FeedForwardNN([1,1],[1,1])
   b_datafile = datafile.encode('UTF-8')
   cdef char *c_datafile = b_datafile
   nn.thisptr = new cn.FeedForwardNN(c_datafile)
   return nn

cdef class FeedForwardNN(object):

   cdef cn.FeedForwardNN *thisptr

   def __cinit__(self, layers, functions):
      if len(layers) != len(functions):
         raise Exception()

      cdef int i

      cdef int *c_layers = <int*>malloc(sizeof(int)*len(layers))
      for i in xrange(len(layers)):
         c_layers[i] = layers[i]

      cdef int *c_functions = <int*>malloc(sizeof(int)*len(layers))
      for i in xrange(len(functions)):
         c_functions[i] = functions[i]

      self.thisptr = new cn.FeedForwardNN(len(layers), c_layers, c_functions)
      free(c_layers)
      free(c_functions)

   def __dealloc__(self):
      if self.thisptr is not NULL:
         del self.thisptr

   def compute(self, x):
      cdef int i
      cdef int *layers = self.thisptr.getLayersSize()
      cdef int layers_num = self.thisptr.getNumOfLayers()

      cdef float *c_x = <float*>malloc(sizeof(float)*len(x))
      for i in xrange(len(x)):
         c_x[i] = x[i]

      cdef float *c_y = <float*>malloc(sizeof(float)*layers[layers_num-1])
      for i in xrange(layers[layers_num-1]):
         c_y[i] = 0

      self.thisptr.compute(c_x, c_y)

      y = []
      for i in xrange(layers[layers_num-1]):
         y.append(c_y[i])

      free(c_x)
      free(c_y)
      return y

   def saveToTxt(self, str filename):
      b_filename = filename.encode('UTF-8')
      cdef char *c_filename = b_filename
      self.thisptr.saveToTxt(c_filename)

cdef class FeedForwardNNTrainer(object):

   cdef cn.FeedForwardNNTrainer *thisptr

   def __cinit__(self):
      self.thisptr = new cn.FeedForwardNNTrainer()

   def __dealloc__(self):
      if self.thisptr is not NULL:
         del self.thisptr

   def selectNet(self, FeedForwardNN net):
      self.thisptr.selectNet(net.thisptr[0])

   def selectTrainingSet(self, LearningSet training_set):
      self.thisptr.selectTrainingSet(training_set.thisptr[0])

   def selectTestSet(self, LearningSet test_set):
      self.thisptr.selectTestSet(test_set.thisptr[0])

   def train(self, **keywords):
      params_order = ["device", "algorithm", "desired_error", "max_epochs",
                      "epochs_between_reports", "learning_rate", "momentum",
                      "shuffle", "error_function"]
      params = {"device": TRAIN_GPU, "algorithm": ALG_BATCH, "learning_rate": 0.7,
            "momentum": 0.0, "max_epochs": 1000, "desired_error": 0.001,
            "epochs_between_reports": 100, "shuffle": True,
            "error_function": ERROR_TANH, "print_type": PRINT_MIN}
      params.update(keywords)
      if params["shuffle"]:
         params["shuffle"] = SHUFFLE_ON
      else:
         params["shuffle"] = SHUFFLE_OFF

      cdef float *c_params = <float*>malloc(sizeof(float)*len(params_order))
      for i in xrange(len(params_order)):
         c_params[i] = params[params_order[i]]

      cdef float res = self.thisptr.train(len(params_order), c_params,
                                          params["print_type"])
      free(c_params)
      return res

class NeuralNetClassifier(object):

  def __init__(self, hidden_layers=[10], activation_functions=[SIGM, SIGM, SIGM], **kwargs):
     self.hidden_layers = hidden_layers
     self.activation_functions = activation_functions
     self.train_params = kwargs
     self.is_symmetric_output = (self.activation_functions[-1] == TANH)

  def fit(self, X, Y):
     self.classes_ = list(np.unique(Y))

     Yn = []
     for y in Y:
       yn = [0.0] * len(self.classes_)
       yn[self.classes_.index(y)] = 1.0
       Yn.append(yn)

     self.ann = FeedForwardNN([len(X[0])] + self.hidden_layers + [len(self.classes_)],
                                  self.activation_functions)
     nn_data = LearningSet(X, Yn)
     trainer = FeedForwardNNTrainer()
     trainer.selectNet(self.ann)
     trainer.selectTrainingSet(nn_data)
     trainer.train(**self.train_params)

  def predict_proba(self, X):
     try:
        len(X[0])
     except TypeError:
        X = [X]
     if self.is_symmetric_output:
        return np.array([[(r+1.0)/2.0 for r in self.ann.compute(x)] for x in X])
     else:
        return np.array([self.ann.compute(x) for x in X])

  def predict(self, X):
     probas = self.predict_proba(X)
     return np.array([self.classes_[p.argmax()] for p in probas])
