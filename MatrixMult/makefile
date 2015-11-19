build: matrixmult.exe
#Do some stuff

clean:
	 rm -f debug/matrixmult*.*

cleanall:
	rm -f debug/*

debug/matrix_mult.obj: src/matrixmult.cu header/matrixmult.h
	nvcc -c src/matrixmult.cu -o debug/matrix_mult.obj

debug/cudatool.obj: src/cudatool.cu header/cudatool.h
	nvcc -c src/cudatool.cu -o debug/cudatool.obj

debug/matrix_main.obj: src/matrix_main.cu
	nvcc -c src/matrix_main.cu -o debug/matrix_main.obj

matrixmult.exe: cleanCMD debug/cudatool.obj debug/matrix_mult.obj debug/matrix_main.obj
	nvcc debug/cudatool.obj debug/matrix_mult.obj debug/matrix_main.obj -o debug/matrixmult.exe

cleanCMD:
	cls #clean WindowsCMD

#make sollte auch debug folder erstellen
