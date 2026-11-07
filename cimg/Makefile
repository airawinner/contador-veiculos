CXX = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra \
           -Dcimg_display=0 \
           -Dcimg_use_jpeg \
           -Dcimg_use_png

LIBS = -ljpeg -lpng -lz

TARGET = build/cimg_preprocess
SOURCE = src/main.cpp

all: $(TARGET)

$(TARGET): $(SOURCE) include/CImg.h
	mkdir -p build
	$(CXX) $(CXXFLAGS) -Iinclude $(SOURCE) -o $(TARGET) $(LIBS)

clean:
	rm -rf build