SRCS := $(shell find . -name "*.cu")
OBJS := $(SRCS:%.cu=%.o) # substitution reference
CC=nvcc
CFLAGS=-std=c++17 -rdc=true
BIN = main
EXT = cu

test: $(BIN)
	./$(BIN) 

$(BIN): $(OBJS)
	$(CC) -o $(BIN) $^ -O3
gdb: main_debug
	gdb main_debug -nx
main_debug: $(SRCS)
	$(CC) -o main_debug $^ -g

%.o: %.cu
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm -f *.o $(BIN)

.PHONY: clean test main_debug
.SILENT: test