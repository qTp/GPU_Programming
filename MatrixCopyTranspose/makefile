LINK_TARGET= tcMatrix.exe
CFLAGS = -Xcompiler -fopenmp -arch=sm_20 -c -o
#-Xcompiler -Wall-> muss in Datei!!! total simple einfach so aufrufen = make > compiler.log 2>&1
#-Xcompiler -DSERIAL
#-Xcompiler -DDEBUG
CC	= nvcc
PROG_PARAM = 4096 4096 500
PROG_PARAM_TEST = 256 256 20

P_DEBUG = ./debug/
P_HEADER = ./header/
P_SRC = ./src/
#################################
# path for generaten		#
# for simplification P_DEBUG	#
#################################
P_GEN = ${P_DEBUG}

files= cudatool transposeCopyMatrixGPU transposeCopyMatrixCPU main

OBJECTS = $(patsubst %, ${P_GEN}%, $(files:=.obj))

.PHONY: test clean time all

all: ${LINK_TARGET}
	### All done! ###

${LINK_TARGET}: ${OBJECTS}
	${CC} -lgomp ${OBJECTS} -o ${P_GEN}$(LINK_TARGET)

${P_GEN}%.obj: ${P_SRC}%.cu ${P_HEADER}%.h
	$(CC) ${CFLAGS} $@ $<

test:
	cuda-memcheck --log-file memCheck.log ${P_GEN}${LINK_TARGET} ${PROG_PARAM_TEST}

time:
	${P_GEN}${LINK_TARGET} ${PROG_PARAM}

clean:
	rm -f ${P_DEBUG}*.* *.log
