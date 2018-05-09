#!/bin/bash
cython tfidf.py --cplus -o tfidf.cpp
clang++ -Ofast -march=native -msse4.2 $(pkg-config --cflags --libs python3) -shared -fPIC tfidf.cpp -o tfidf.so
python test.py
