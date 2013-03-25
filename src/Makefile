
INSTALL_LIB_DIR := /usr/local/lib
INSTALL_INCLUDE_DIR := /usr/local/include

CUDA_LIB_DIR := /usr/local/cuda/lib64
CUDA_INCLUDE_DIR := /usr/local/cuda/include

###

# Cuda source files (compiled with cudacc)
CUFILES		:= CudaActivationFunctions.cu CudaErrorFunctions.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= FeedForwardNN.cpp LearningSet.cpp FeedForwardNNTrainer.cpp \
             ActivationFunctions.cpp ErrorFunctions.cpp GAFeedForwardNN.cpp \
             FloatChromosome.cpp

OBJS := $(patsubst %.cpp,%.cpp.o,$(notdir $(CCFILES)))
OBJS += $(patsubst %.cu,%.cu.o,$(notdir $(CUFILES)))

INCLUDE_DIR = ../include

LIBS := -L$(CUDA_LIB_DIR) -lcuda -lcublas
INCLUDES := -I$(CUDA_INCLUDE_DIR) -I$(INCLUDE_DIR) -I$(INCLUDE_DIR)/libcudann

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

install: libcudann.a libcudann.so
	cp -t $(INSTALL_LIB_DIR) libcudann.a libcudann.so
	cp -rt $(INSTALL_INCLUDE_DIR) $(INCLUDE_DIR)/* 

uninstall:
	rm $(INSTALL_LIB_DIR)/libcudann.a
	rm $(INSTALL_LIB_DIR)/libcudann.so
	rm -rf $(INSTALL_INCLUDE_DIR)/libcudann

.PHONY: all clean
