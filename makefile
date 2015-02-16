PYINC = /usr/include/python3.4
PYLIB = /usr/local/bin
CFLAGS = -O3 -fopenmp
all: _wavenet.so wavenet.py _wavelet.so wavelet.py

# обертка + действительный класс
_wavenet.so: wavenet_wrap.o net.o train.o
	      g++ $(CFLAGS) -shared wavenet_wrap.o net.o train.o -L $(PYLIB) -o $@
# генерирует модуль обертки класса
wavenet_wrap.o: wavenet_wrap.cxx
		g++ $(CFLAGS) wavenet_wrap.cxx -c -g -fPIC -I $(PYINC)
wavenet_wrap.cxx: wavenet.i
		  swig -c++ -python wavenet.i
wavenet.py: wavenet.i
	    swig -c++ -python wavenet.i

# программный код обертки класса C++
net.o:
	  g++ $(CFLAGS) ../wavenet/net.cpp -c -g -fPIC -Wno-deprecated
train.o:
	  g++ $(CFLAGS) ../wavenet/train.cpp -c -g -fPIC -Wno-deprecated


# обертка + действительный класс
_wavelet.so: wavelet_wrap.o  wavelet.o
	      g++ $(CFLAGS) -shared wavelet_wrap.o wavelet.o -L $(PYLIB) -o $@
# генерирует модуль обертки класса
wavelet_wrap.o: wavelet_wrap.cxx
		g++ $(CFLAGS) wavelet_wrap.cxx -c -g -fPIC -I $(PYINC)
wavelet_wrap.cxx: wavelet.i
		  swig -c++ -python wavelet.i
wavelet.py: wavelet.i
	    swig -c++ -python wavelet.i

# программный код обертки класса C++
wavelet.o:
	  g++ $(CFLAGS) ../wavenet/wavelet.cpp -c -g -fPIC -Wno-deprecated


clean:
	rm -f *.pyc *.o *.so *.py *.cxx

