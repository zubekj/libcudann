PYTHON_INSTALL_DIR = /usr/lib/python2.7/site-packages

all: pycudann.so

pycudann.so:
	python2 setup.py build_ext --inplace

clean:
	rm pycudann.so
	rm -rf build
	rm pycudann.cpp

install: pycudann.so
	cp -t $(PYTHON_INSTALL_DIR) pycudann.so

uninstall:
	rm $(PYTHON_INSTALL_DIR)/pycudann.so

.PHONY: all clean install uninstall
