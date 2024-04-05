EXECUTABLE := generator
CU_FILES   := 
CU_DEPS    :=
CC_FILES   := generator.cpp


CXX=g++ -m64
CXXFLAGS=-O3 -Wall

all: $(EXECUTABLE)

$(EXECUTABLE): $(CC_FILES)
	$(CXX) $(CXXFLAGS) $(CC_FILES) -o $(EXECUTABLE)

clean:
	rm $(EXECUTABLE) *.txt