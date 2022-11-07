DIRBIN = ./bin
TARGET = $(DIRBIN)/$(EXE)
EXE = denmat_dynm_v4.5.7
SRC_DIRS = ./src_v4.5.7

all: $(TARGET)

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s)
OBJS := $(addsuffix .o,$(basename $(SRCS)))
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS = $(INC_FLAGS) -MMD -MP

$(TARGET): $(OBJS)
	$(CC) $(CPPFLAGS) $(LDFLAGS) $(OBJS) -o $@ $(LOADLIBES) $(LDLIBS)

$(OBJS) : $(SRC_DIRS)/%.o: $(SRC_DIRS)/%.cpp
	$(CC) $(CPPFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	rm -rf $(TARGET) $(OBJS) $(DEPS)

-include $(DEPS) ./make.inc

rm_d_files:
	$(shell rm $(SRC_DIRS)/*.d)
