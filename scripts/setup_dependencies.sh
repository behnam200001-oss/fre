#!/bin/bash

# Advanced Bitcoin Miner CUDA - Dependency Setup Script
# Usage: ./scripts/setup_dependencies.sh

set -e

echo "🚀 Setting up dependencies for Advanced Bitcoin Miner CUDA..."

# بررسی وجود CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA not found. Please install CUDA Toolkit 11.0 or higher"
    exit 1
fi

# بررسی وجود CMake
if ! command -v cmake &> /dev/null; then
    echo "📦 Installing CMake..."
    sudo apt-get update
    sudo apt-get install -y cmake
fi

# ایجاد دایرکتوری‌های لازم
mkdir -p third_party
mkdir -p build
mkdir -p outputs/{logs,results}
mkdir -p data/addresses

# دانلود و کامپایل secp256k1
echo "📥 Downloading secp256k1..."
cd third_party
if [ ! -d "secp256k1" ]; then
    git clone https://github.com/bitcoin-core/secp256k1.git
    cd secp256k1
    ./autogen.sh
    ./configure --enable-module-recovery --enable-experimental --enable-module-ecdh
    make
    cd ..
fi

# بررسی وجود OpenSSL
if ! pkg-config --exists openssl; then
    echo "📦 Installing OpenSSL development packages..."
    sudo apt-get install -y libssl-dev
fi

# نصب Google Test برای تست‌ها
echo "📥 Setting up Google Test..."
if [ ! -d "googletest" ]; then
    git clone https://github.com/google/googletest.git
    cd googletest
    mkdir build && cd build
    cmake ..
    make
    sudo make install
    cd ../..
fi

# ایجاد فایل پیکربندی پیش‌فرض
cd ..
if [ ! -f "config/miner.conf" ]; then
    cp config/miner.conf.example config/miner.conf
    echo "📝 Created default configuration file"
fi

echo "✅ Dependency setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Review config/miner.conf for your settings"
echo "2. Run: make release"
echo "3. Run: ./bin/advanced_bitcoin_miner"