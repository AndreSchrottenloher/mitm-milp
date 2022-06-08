all: haraka


.PHONY: clean

clean:
	rm -f *.o

%.o: %.cpp
	g++ -o $@ -c $< -Wall -pedantic -std=c++11 -O2 -g -lpthread
	
haraka:
	g++ haraka.cpp  -o $@  -Wall -Wno-ignored-attributes -pedantic -std=c++11 -O2 -g -lpthread -msse4.1 -march=native
	./haraka


