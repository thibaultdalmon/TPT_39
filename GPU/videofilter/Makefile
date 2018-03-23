EXE=videofilter
OCLLIBSDIR=/opt/ComputeLibrary/build/
OCLINCSDIR=/opt/ComputeLibrary/include/
LDLIBS = -L${OCLLIBSDIR} -larm_compute -larm_compute_core -lOpenCL `pkg-config --libs --cflags opencv`
CPPFLAGS = -g -Wno-deprecated-declarations -Wall -DARCH_ARM -Wextra -Wno-unused-parameter -pedantic -Wdisabled-optimization -Wformat=2 -Winit-self -Wstrict-overflow=2 -Wswitch-default -fpermissive -std=gnu++11 -Wno-vla -Woverloaded-virtual -Wctor-dtor-privacy -Wsign-promo -Weffc++ -Wno-format-nonliteral -Wno-overlength-strings -Wno-strict-overflow -Wno-implicit-fallthrough -Wlogical-op -Wnoexcept -Wstrict-null-sentinel -march=armv7-a -mthumb -mfpu=neon -mfloat-abi=hard -O3 -ftree-vectorize -fstack-protector-strong -DARM_COMPUTE_CL -I${OCLINCSDIR} -I.. `pkg-config --libs opencv`
CC=g++
SRCS=videofilter.cpp
all:${EXE}
${EXE}: ${SRCS}


profile:
	./${EXE}
	gprof ./${EXE}  gmon.out > prof.txt
clean:
	rm -rf ${EXE} *.o
