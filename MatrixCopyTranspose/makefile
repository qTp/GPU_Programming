
LINK_TARGET= tcMatrix.exe
CFLAGS = -c -Xcompiler -DDEBUG  -o #-Xcompiler -Wall -> muss in Datei!!!
CC	= nvcc
PROG_PARAM = 2048 2048 16

P_DEBUG = ./debug/
P_HEADER = ./header/
P_SRC = ./src/
###############################
# path for generaten					#
# for simplification P_DEBUG	#
###############################
P_GEN = ${P_DEBUG}

files= cudatool transposeCopyMatrixGPU transposeCopyMatrixCPU main

OBJECTS = $(patsubst %, ${P_GEN}%, $(files:=.obj))

.PHONY: test clean time all

all: ${LINK_TARGET}
	### All done! ###

${LINK_TARGET}: ${OBJECTS}
	${CC} ${OBJECTS} -o ${P_GEN}$(LINK_TARGET)

${P_GEN}%.obj: ${P_SRC}%.cu ${P_HEADER}%.h
	$(CC) ${CFLAGS} $@ $<

test:
	cuda-memcheck --log-file memCheck.log ${P_DEBUG}${LINK_TARGET} ${PROG_PARAM}

time:
	${P_DEBUG}${LINK_TARGET} ${PROG_PARAM}

clean:
	rm -f ${P_DEBUG}*.* *.log
