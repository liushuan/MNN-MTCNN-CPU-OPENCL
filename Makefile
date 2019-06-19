CXX = g++
OBJS = $(SRCS:.cpp = .o)
SRCS = $(wildcard *.cpp)


MNN = /root/work_space/software/MNN
OPENCV = /root/work_space/software/opencv3.3.0
INCPATH =      -I${MNN}/include \
                -I${OPENCV}/include

LIBS =  -lopencv_core -lopencv_highgui -lopencv_imgproc  -lopencv_imgcodecs -lopencv_video -lopencv_videoio -lopencv_videostab  -lMNN -lMNN_CL -linterfaceAFR  -ldl
LIBPATH = -L${OPENCV}/lib -L${MNN}/build -L/root/work_space/source_code/mnn/interface_AFR


CXXFLAGS = -std=c++11
OUTPUT = AFR_test          #输出程序名称

#%:%.cpp
#       $(CXX) $(INCPATH)$(LIBPATH) $^ ${NCNN}/build/install/lib/libncnn.a $(LIBS) -o $@

all:$(OUTPUT)
$(OUTPUT) : $(OBJS)
	$(CXX) $^  -o $@ $(INCPATH) $(CXXFLAGS) $(LIBPATH) $(LIBS)  -Wl,-R${MNN}/build -Wl,-R/root/work_space/source_code/mnn/interface_AFR
.PHONY:clean
clean:
	rm -rf *.out *.o $(OUTPUT)  #清除中间文件及生成文件

