EXECUTABLE := uwu

CU_FILES   := grayscale.cu convolve.cu harris.cu brief.cu matching.cu

CU_DEPS    :=

CC_FILES   := main.cpp

###########################################################

OBJDIR=objs
CXX=g++
CXXFLAGS=-O3 -Wall -I/usr/include/opencv4 -I/usr/local/cuda-10.2/include -fopenmp
LDFLAGS= -L/usr/lib -lopencv_core -lopencv_highgui -lopencv_videoio -L/usr/local/cuda-10.2/lib64 -lcudart -lopencv_imgproc
NVCC=nvcc
NVCCFLAGS=-O3 -ccbin /usr/bin/g++ -arch=sm_53

OBJS=$(OBJDIR)/grayscale.o $(OBJDIR)/convolve.o $(OBJDIR)/harris.o $(OBJDIR)/brief.o $(OBJDIR)/matching.o

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
