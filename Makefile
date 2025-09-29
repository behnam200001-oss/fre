# Advanced Bitcoin Miner CUDA - Professional Makefile (Fixed)
CC := nvcc
CXX := g++
CFLAGS := -O3 -std=c++17 -arch=sm_80 -Xptxas -O3,-v -lineinfo
CFLAGS += --use_fast_math -Xcompiler -fopenmp -DNDEBUG
CFLAGS += -Xcompiler -fPIC --expt-relaxed-constexpr --expt-extended-lambda
CFLAGS += -Xcompiler -Wno-deprecated-declarations

# مسیرهای include
CFLAGS += -I./src -I./src/core -I./src/crypto -I./src/bloom
CFLAGS += -I./src/gpu -I./src/address -I./src/storage
CFLAGS += -I./src/monitoring -I./src/utils -I./src/config
CFLAGS += -I./kernels -I./third_party/secp256k1/include
CFLAGS += -I./third_party/secp256k1/src
CFLAGS += -DGPU_BLOOM_OPTIMIZED -DMULTI_GPU_SUPPORT -DADVANCED_OPTIMIZATIONS

# مدیریت خطاهای CUDA
CFLAGS += -Xcudafe --diag_suppress=code_is_unreachable
CFLAGS += -Xcudafe --diag_suppress=initialization_not_reachable

LDFLAGS := -L/usr/lib/x86_64-linux-gnu -L/usr/local/cuda/lib64
LIBS := -lcrypto -lssl -lpthread -ldl -lrt -fopenmp
CUDA_LIBS := -lcudart -lcuda -lcublas -lcurand

SRC_DIRS := src core crypto bloom gpu address storage monitoring utils config
VPATH := $(addprefix src/, $(SRC_DIRS)) kernels

# جمع‌آوری تمام فایل‌های منبع
SRCS_CU := $(wildcard src/*.cu) $(wildcard src/*/*.cu) $(wildcard kernels/*.cu)
SRCS_CPP := $(wildcard src/*.cpp) $(wildcard src/*/*.cpp)
SRCS_C := $(wildcard src/*.c) $(wildcard src/*/*.c)

OBJS_CU := $(SRCS_CU:.cu=.o)
OBJS_CPP := $(SRCS_CPP:.cpp=.o)
OBJS_C := $(SRCS_C:.c=.o)
OBJS := $(OBJS_CU) $(OBJS_CPP) $(OBJS_C)

DEPS := $(OBJS:.o=.d)

TARGET := bin/advanced_bitcoin_miner

.PHONY: all release debug profile clean distclean test help

all: release

release: CFLAGS += -DOPTIMIZED -DNDEBUG
release: dirs $(TARGET)

debug: CFLAGS += -G -g -DDEBUG -DGPU_DEBUG=1
debug: CFLAGS += -Xptxas -O0
debug: dirs $(TARGET)

profile: CFLAGS += -pg -DPROFILING
profile: dirs $(TARGET)

$(TARGET): $(OBJS)
	@echo "🔗 Linking target..."
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LIBS) $(CUDA_LIBS) $(LDFLAGS)
	@echo "✅ Build completed: $(TARGET)"

%.o: %.cu
	@echo "🔨 Compiling CUDA $<..."
	$(CC) $(CFLAGS) -M -MT $@ -MF $(@:.o=.d) $<
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	@echo "🔨 Compiling C++ $<..."
	$(CXX) $(CFLAGS) -c $< -o $@

%.o: %.c
	@echo "🔨 Compiling C $<..."
	$(CXX) $(CFLAGS) -c $< -o $@

dirs:
	@mkdir -p bin outputs/logs outputs/results data/addresses tests
	@mkdir -p build/src build/kernels

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -f $(OBJS) $(DEPS) $(TARGET)

distclean: clean
	rm -rf outputs/* data/* logs/*

test: debug
	@echo "🧪 Running tests..."
	./scripts/run_tests.sh

help:
	@echo "Advanced Bitcoin Miner CUDA - Build System"
	@echo "Available targets:"
	@echo "  release   - Optimized release build"
	@echo "  debug     - Debug build with symbols"
	@echo "  profile   - Profiling build"
	@echo "  test      - Run test suite"
	@echo "  clean     - Clean build artifacts"
	@echo "  distclean - Clean everything"

-include $(DEPS)