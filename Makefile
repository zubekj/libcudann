# Add source files here
# Cuda source files (compiled with cudacc)
CUFILES		:= CudaActivationFunctions.cu CudaErrorFunctions.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= FeedForwardNN.cpp LearningSet.cpp FeedForwardNNTrainer.cpp ActivationFunctions.cpp ErrorFunctions.cpp GAFeedForwardNN.cpp FloatChromosome.cpp

USECUBLAS	:= 1

OBJS := $(patsubst %.cpp,%.cpp.o,$(notdir $(CCFILES)))
OBJS += $(patsubst %.cu,%.cu.o,$(notdir $(CUFILES)))

LIBDIR := /opt/cuda/lib64
INCLUDEDIR := /opt/cuda-toolkit/include

LIBS := -L$(LIBDIR) -lcuda -lcublas
INCLUDES := -I$(INCLUDEDIR)

CXX := g++
NVCC := nvcc

all: cudann_test libcudann.a libcudann.so

%.cu.o : %.cu
	$(NVCC) $(INCLUDES) -Xcompiler -fPIC -o $@ -c $<

%.cpp.o : %.cpp
	$(CXX) $(INCLUDES) -fPIC -o $@ -c $<

libcudann.a: $(OBJS)
	ar -cvq libcudann.a $(OBJS)	

libcudann.so: $(OBJS)
	gcc -shared -Wl,-soname,libcudann.so \
	       -o libcudann.so $(OBJS) $(LIBS)

cudann_test: main.cpp.o libcudann.a
	$(CXX) main.cpp.o libcudann.a $(LIBS) -o cudann_test
	
clean:
	rm *.o cudann_test libcudann.a libcudann.so

.PHONY: all clean
