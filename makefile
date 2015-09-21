PYINC = /usr/include/python3.4
PYLIB = /usr/local/bin
CFLAGS = -O3 -fopenmp
CMODS = _cregister.so

all: _wavenet.so  _wavelet.so cregister.o $(CMODS)

# обертка + действительный класс
_wavenet.so: wavenet_wrap.o net.o train.o
	      g++ $(CFLAGS) -shared wavenet_wrap.o net.o train.o -L $(PYLIB) -o $@
# генерирует модуль обертки класса
wavenet_wrap.o: wavenet_wrap.cxx
		g++ $(CFLAGS) wavenet_wrap.cxx -c -g -fPIC -I $(PYINC)
wavenet_wrap.cxx: ../interfaces/wavenet.i
		swig -outdir . -o ./wavenet_wrap.cxx -c++ -python ../interfaces/wavenet.i

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
wavelet_wrap.cxx: ../interfaces/wavelet.i
		swig -outdir . -o ./wavelet_wrap.cxx -c++ -python  ../interfaces/wavelet.i

# программный код обертки класса C++
wavelet.o:
	  g++ $(CFLAGS) ../wavenet/wavelet.cpp -c -g -fPIC -Wno-deprecated

# обертка + действительный класс
$(CMODS): cregister_wrap.o cregister.o
	      g++ $(CFLAGS) -shared cregister_wrap.o cregister.o  -L $(PYLIB) -o $@
# генерирует модуль обертки класса
cregister_wrap.o: cregister_wrap.cxx
		g++ $(CFLAGS) cregister_wrap.cxx -c -g -fPIC -I $(PYINC) 
cregister_wrap.cxx: ../interfaces/cregister.i
		  swig -c++ -python -outdir . -o ./cregister_wrap.cxx ../interfaces/cregister.i

# программный код обертки класса C++
cregister.o:
	  g++ $(CFLAGS) ../wavenet/cregister.cpp -c -g -fPIC -Wno-deprecated

clean:
	rm -f *.pyc *.o *.so *.py *.cxx

