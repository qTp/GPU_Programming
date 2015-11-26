
LINK_TARGET=tcMatrix.exe
CFLAGS = -c -Xcompiler -DDEBUG -o
CC = nvcc
PROG_PARAM=

OBJECTS= cudatool.obj transposeCopyMatrix.obj main.obj

P_DEBUG = ./debug/
P_HEADER = ./header/
P_SRC = ./src/

.PHONY: test clean

all: ${LINK_TARGET}
	### All done! ###

${LINK_TARGET}: ${OBJECTS}
	${CC} ${OBJECTS} -o debug/$(LINK_TARGET)

%.obj: ${P_SRC}%.cu ${P_HEADER}%.h
	$(CC) ${CFLAGS} $@ $<

test:
	cuda-memcheck --log-file memCheck.log debug/${LINK_TARGET}${PROG_PARAM}

time:
	debug/${LINK_TARGET} ${PROG_PARAM}

clean:
	rm -f debug/*.* ./*.obj
