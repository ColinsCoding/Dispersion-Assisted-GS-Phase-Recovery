NVCC     := nvcc
ARCH     := sm_86          # RTX 30xx/40xx; change to sm_75 for Turing, sm_89 for Ada
CFLAGS   := -O3 -arch=$(ARCH) -lineinfo
LIBS     := -lcufft
TARGET   := tdgs_anneal

.PHONY: build run clean

build:
	$(NVCC) $(CFLAGS) $(LIBS) tdgs_anneal.cu -o $(TARGET)

run: build
	./$(TARGET)

# detect GPU arch automatically if nvcc is available
auto-arch:
	@$(NVCC) --query-gpu-info 2>/dev/null | grep "Compute Capability" | \
	  awk '{printf "sm_%s%s\n", $$3, $$4}' | head -1

clean:
	rm -f $(TARGET) $(TARGET).exe *.o
