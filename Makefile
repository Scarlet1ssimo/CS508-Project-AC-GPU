SRCS := $(shell find . -name "*.cu")
HEADERS := $(shell find . -name "*.h")
OBJS := $(SRCS:%.cu=%.o) # substitution reference
CC=nvcc
CFLAGS=-rdc=true -O3
BIN = main
EXT = cu

test: $(BIN)
	./$(BIN) 

$(BIN): $(OBJS)
	$(CC) -o $(BIN) $^
gdb: main_debug
	gdb main_debug -nx
main_debug: $(SRCS)
	$(CC) -o main_debug $^ -g

%.o: %.cu $(HEADERS)
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm -f *.o $(BIN) main_debug

.PHONY: clean test main_debug rai
.SILENT: test