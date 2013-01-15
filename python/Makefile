all: pycudann.so

pycudann.so:
	python2 setup.py build_ext --inplace

clean:
	rm pycudann.so
	rm -rf build
	rm pycudann.cpp

.PHONY: all
