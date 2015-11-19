MKDIR_P = mkdir -p
DEBUG_DIR = debug
OUT_O_DIR = ${DEBUG_DIR}/objects
SRC_DIR = src
HEADER_DIR = header
CC = nvcc
CFLAGS = -c -o
LFLAGS =
LINK_TARGET = MatrixMultTool.exe

OBJECTS = $(OUT_O_DIR)/cudatool.obj \
 					$(OUT_O_DIR)/matrix_mult.obj \
					$(OUT_O_DIR)/main.obj

.PHONY: directories cleanCMD cleanall clean

all: cleanCMD directories $(LINK_TARGET)
	### All done! ###

clean:
	 rm -f ${DEBUG_DIR}/*.*

cleanall:
	rm -rf ${DEBUG_DIR}

directories: ${DEBUG_DIR} ${OUT_O_DIR}

${OUT_O_DIR}:
	${MKDIR_P} ${OUT_O_DIR}

${DEBUG_DIR}:
		${MKDIR_P} ${DEBUG_DIR}

${OUT_O_DIR}/matrix_mult.obj: ${SRC_DIR}/matrixmult.cu  ${HEADER_DIR}/matrixmult.h ${HEADER_DIR}/cudatool.h
	${CC} $(CFLAGS) ${OUT_O_DIR}/matrix_mult.obj ${SRC_DIR}/matrixmult.cu

${OUT_O_DIR}/cudatool.obj: ${SRC_DIR}/cudatool.cu  ${HEADER_DIR}/cudatool.h
	${CC} $(CFLAGS) ${OUT_O_DIR}/cudatool.obj ${SRC_DIR}/cudatool.cu

${OUT_O_DIR}/main.obj: ${SRC_DIR}/main.cu ${HEADER_DIR}/matrixmult.h ${HEADER_DIR}/cudatool.h
	${CC} $(CFLAGS) ${OUT_O_DIR}/main.obj ${SRC_DIR}/main.cu

$(LINK_TARGET): $(OBJECTS)
	${CC} $(LFLAGS) $(OBJECTS) -o ${DEBUG_DIR}/$(LINK_TARGET)

cleanCMD:
	cls #clean WindowsCMD
