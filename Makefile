EXECUTABLE := uwu

CU_FILES   := grayscale.cu

CU_DEPS    :=

CC_FILES   := main.cpp

###########################################################

OBJDIR=objs
CXX=g++
CXXFLAGS=-O3 -Wall -I/usr/include/opencv4 -I/usr/local/cuda-10.2/include
LDFLAGS= -L/usr/lib -lopencv_core -lopencv_highgui -lopencv_videoio -L/usr/local/cuda-10.2/lib64 -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/g++

OBJS=$(OBJDIR)/grayscale.o

all: $(EXECUTABLE)

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) main.cpp -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
