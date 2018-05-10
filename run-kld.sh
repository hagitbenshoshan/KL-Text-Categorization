#!/bin/bash
cython kld.py --cplus -o kld.cpp
clang++ -Ofast -march=native -msse4.2 $(pkg-config --cflags --libs python3) -shared -fPIC kld.cpp -o kld.so
python test-kld.py
