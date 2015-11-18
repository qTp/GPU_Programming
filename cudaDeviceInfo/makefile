###need to add targets
all: compile

clean:
	rm -f debug/deviceTool.*

cleanall:
	rm -f debug/*

compile: cleanCMD debug/cudaTool.obj debug/main.obj
	nvcc debug/main.obj debug/cudaTool.obj -o debug/deviceTool.exe

debug/cudaTool.obj: src/cudatool.cu header/cudatool.h
	nvcc -c src/cudatool.cu -o debug/cudaTool.obj

debug/main.obj: src/main.cu
	nvcc -c src/main.cu -o debug/main.obj

cleanCMD:
	cls #clean the WindowsCMD
