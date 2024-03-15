COM=g++
VER=-std=c++2a

output: main.o gbm.o net.o ddpg.o
	$(COM) $(VER) main.o gbm.o net.o ddpg.o -o exec
	rm *.o

main.o: ./src/main.cpp
	$(COM) $(VER) -c ./src/main.cpp

gbm.o: ./src/gbm.cpp
	$(COM) $(VER) -c ./src/gbm.cpp

net.o: ./src/net.cpp
	$(COM) $(VER) -c ./src/net.cpp

ddpg.o: ./src/ddpg.cpp
	$(COM) $(VER) -c ./src/ddpg.cpp