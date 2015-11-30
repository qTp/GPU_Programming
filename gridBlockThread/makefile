all: gbt.exe

gbt.exe: ./src/main.cu
	nvcc -o gbt.exe ./src/main.cu

clean:
	rm -f gbt.*
