HEADER = de.h
TARGET = lshade_exe

OBJS := $(patsubst %.cc,%.o,$(wildcard *.cc))
## CC = g++
CC = g++-6
OPTION = -Ofast -march=native

$(TARGET): $(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(OPTION) -lm 

%.o: %.cc $(HEADER)
	$(CC) $(CFLAGS) $(OPTION)  -c $<

# 
# jTranspose: jTranspose.cpp
#        g++ jTranspose.cpp -o jTranspose -std=c++11


clean:
	rm -rf *.o
