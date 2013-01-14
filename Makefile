# Add source files here
EXECUTABLE	:= execute
# Cuda source files (compiled with cudacc)
CUFILES		:= CudaActivationFunctions.cu CudaErrorFunctions.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= main.cpp FeedForwardNN.cpp LearningSet.cpp FeedForwardNNTrainer.cpp ActivationFunctions.cpp ErrorFunctions.cpp GAFeedForwardNN.cpp FloatChromosome.cpp

USECUBLAS	:= 1

OBJS := $(patsubst %.cpp,%.cpp.o,$(notdir $(CCFILES)))
OBJS += $(patsubst %.cu,%.cu.o,$(notdir $(CUFILES)))

LIBDIR := /opt/cuda/lib64
INCLUDEDIR := /opt/cuda-toolkit/include

LIBS := -L$(LIBDIR) -lcuda -lcublas
INCLUDES := -I$(INCLUDEDIR)

CXX := g++
NVCC := nvcc

%.cu.o : %.cu
	$(NVCC) $(INCLUDES) -o $@ -c $<

%.cpp.o : %.cpp
	$(CXX) $(INCLUDES) -o $@ -c $<

$(EXECUTABLE): $(OBJS)
	$(CXX) $(OBJS) $(LIBS) -o $(EXECUTABLE)

clean:
	rm *.o $(EXECUTABLE)

.PHONY: clean
