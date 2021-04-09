BINDIR=bin
OBJDIR=objs
main:
	g++ -O3 -std=c++11 -Wall -I/usr/include/opencv4 simple_camera.cpp -L/usr/lib -lopencv_core -lopencv_highgui -lopencv_videoio -o $(BINDIR)/simple_camera

