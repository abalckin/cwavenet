all: newn

newn: main.o core.o
	g++ -pg -std=c++11 core.o newn.o -o newn

main.o: core.o
	g++ -pg -std=c++11 -c ../newn.cpp 

core.o:
	g++ -pg -std=c++11 -c ../../wavenet/core.cpp -o core.o
clean:
	rm -rf *.o newn
#optim.o:
	#g++ -c ../dlib/optimization.h -o optimization.h.gch
