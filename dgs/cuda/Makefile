NVCC        := nvcc
CUDA_PATH   ?= /usr/local/cuda
INCLUDES    := -I$(CUDA_PATH)/include
LIB_PATH    := -L$(CUDA_PATH)/lib64
CFLAGS      := -O3 -Xcompiler -fPIC
LIBS        := -lcufft -lcudart

SRC_DIR     := .
OBJ_DIR     := build
SRCS        := $(SRC_DIR)/dispersion.cu $(SRC_DIR)/forward_model.cu
OBJS        := $(OBJ_DIR)/dispersion.o $(OBJ_DIR)/forward_model.o
LIB_NAME    := libdgs_forward.a

all: $(LIB_NAME)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(LIB_NAME): $(OBJS)
	$(AR) rcs $@ $^

clean:
	rm -rf $(OBJ_DIR) $(LIB_NAME)

.PHONY: all clean
