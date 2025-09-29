#!/bin/bash

# Advanced Bitcoin Miner CUDA - Build Script
# Usage: ./scripts/build_release.sh [debug|release|profile]

set -e

BUILD_TYPE="${1:-release}"
BUILD_DIR="build_${BUILD_TYPE}"
COMPILE_JOBS=$(nproc)

echo "🏗️ Building Advanced Bitcoin Miner CUDA ($BUILD_TYPE)..."

# بررسی وجود dependencies
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA compiler (nvcc) not found"
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    echo "❌ CMake not found"
    exit 1
fi

# ایجاد دایرکتوری build
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# تنظیمات CMake بر اساس نوع build
case $BUILD_TYPE in
    "debug")
        CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Debug -DGPU_DEBUG=ON"
        ;;
    "release")
        CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release -DOPTIMIZE=ON"
        ;;
    "profile")
        CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=RelWithDebInfo -DPROFILING=ON"
        ;;
    *)
        echo "❌ Unknown build type: $BUILD_TYPE"
        echo "Usage: $0 [debug|release|profile]"
        exit 1
        ;;
esac

# اجرای CMake
echo "📋 Configuring with CMake..."
cmake .. $CMAKE_FLAGS

# کامپایل
echo "🔨 Compiling with $COMPILE_JOBS jobs..."
make -j$COMPILE_JOBS

# بررسی فایل خروجی
if [ -f "./advanced_bitcoin_miner" ]; then
    echo "✅ Build completed successfully!"
    echo "📦 Binary location: $(pwd)/advanced_bitcoin_miner"
    
    # کپی به دایرکتوری bin
    cd ..
    mkdir -p bin
    cp "$BUILD_DIR/advanced_bitcoin_miner" "bin/advanced_bitcoin_miner_$BUILD_TYPE"
    echo "📁 Copied to: bin/advanced_bitcoin_miner_$BUILD_TYPE"
else
    echo "❌ Build failed - binary not found"
    exit 1
fi

# اجرای تست‌ها در حالت debug
if [ "$BUILD_TYPE" = "debug" ]; then
    echo "🧪 Running unit tests..."
    if [ -f "$BUILD_DIR/tests/unit_tests" ]; then
        ./$BUILD_DIR/tests/unit_tests
    else
        echo "⚠️ Unit tests not found, skipping..."
    fi
fi

echo ""
echo "🚀 Build process completed!"
echo "💡 Next: Run './bin/advanced_bitcoin_miner_$BUILD_TYPE' to start mining"