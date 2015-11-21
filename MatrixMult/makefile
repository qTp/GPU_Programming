MKDIR_P = mkdir -p
DEBUG_DIR = ./debug
OUT_O_DIR = ${DEBUG_DIR}/objects
SRC_DIR = ./src
HEADER_DIR = ./header
CC = nvcc
CFLAGS = -c -o
LFLAGS =
LINK_TARGET = MatrixMultTool.exe

MAIN = main
CUDA_TOOL = cudatool
MM_CPU = matrixmult_cpu
MM_GPU = matrixmult_gpu

OBJECTS = $(OUT_O_DIR)/cudatool.obj \
 					$(OUT_O_DIR)/$(MM_CPU).obj \
					$(OUT_O_DIR)/$(MM_GPU).obj \
					$(OUT_O_DIR)/main.obj

.PHONY: directories cleanCMD cleanall clean test debug setFlags

all: cleanCMD directories $(LINK_TARGET)
	### All done! ###

clean:
	 rm -f ${DEBUG_DIR}/*.* ${OUT_O_DIR}/*.*

cleanall:
	rm -rf ${DEBUG_DIR}  ${OUT_O_DIR} *.pdb

directories: ${DEBUG_DIR} ${OUT_O_DIR}

${OUT_O_DIR}:
	${MKDIR_P} ${OUT_O_DIR}

${DEBUG_DIR}:
		${MKDIR_P} ${DEBUG_DIR}

${OUT_O_DIR}/$(MM_CPU).obj: ${SRC_DIR}/$(MM_CPU).cu  ${HEADER_DIR}/$(MM_CPU).h ${HEADER_DIR}/${CUDA_TOOL}.h
	${CC} $(CFLAGS) ${OUT_O_DIR}/$(MM_CPU).obj ${SRC_DIR}/$(MM_CPU).cu

${OUT_O_DIR}/$(MM_GPU).obj: ${SRC_DIR}/$(MM_GPU).cu  ${HEADER_DIR}/$(MM_GPU).h ${HEADER_DIR}/${CUDA_TOOL}.h
	${CC} $(CFLAGS) ${OUT_O_DIR}/$(MM_GPU).obj ${SRC_DIR}/$(MM_GPU).cu

${OUT_O_DIR}/cudatool.obj: ${SRC_DIR}/${CUDA_TOOL}.cu  ${HEADER_DIR}/${CUDA_TOOL}.h
	${CC} $(CFLAGS) ${OUT_O_DIR}/${CUDA_TOOL}.obj ${SRC_DIR}/${CUDA_TOOL}.cu

${OUT_O_DIR}/main.obj: ${SRC_DIR}/${MAIN}.cu ${HEADER_DIR}/${MAIN}.h \
	 										${HEADER_DIR}/$(MM_CPU).h ${HEADER_DIR}/$(MM_GPU).h \
											${HEADER_DIR}/${CUDA_TOOL}.h
	${CC} $(CFLAGS) ${OUT_O_DIR}/${MAIN}.obj ${SRC_DIR}/${MAIN}.cu

$(LINK_TARGET): $(OBJECTS)
	${CC} $(LFLAGS) $(OBJECTS) -o ${DEBUG_DIR}/$(LINK_TARGET)

cleanCMD:
	cls #clean WindowsCMD

#debug: | setFlags ${LINK_TARGET}

#setFlags:
#	CFLAGS :=  -g -G $(CFLAGS)

test:
	cuda-memcheck --log-file memCheck.log debug\MatrixMultTool.exe 100 16 16
