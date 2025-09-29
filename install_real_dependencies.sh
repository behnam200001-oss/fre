#!/bin/bash

# Advanced Bitcoin Miner - Real Dependencies Installation Script
set -e

echo "🔧 Installing real dependencies for Advanced Bitcoin Miner..."

# بروزرسانی سیستم
sudo apt-get update

# نصب وابستگی‌های سیستم
sudo apt-get install -y \
    build-essential \
    cmake \
    nvidia-cuda-toolkit \
    libssl-dev \
    libgtest-dev \
    libomp-dev \
    pkg-config \
    automake \
    libtool

# کامپایل secp256k1 با قابلیت‌های مورد نیاز
echo "📦 Building secp256k1 library..."
cd third_party/secp256k1
./autogen.sh
./configure --enable-module-recovery --enable-experimental --enable-module-ecdh
make
sudo make install
cd ../..

# تنظیم متغیرهای محیطی برای کتابخانه‌ها
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

# اضافه کردن به bashrc برای sessions آینده
echo "export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:\$PKG_CONFIG_PATH" >> ~/.bashrc

echo "✅ Real dependencies installed successfully!"
echo "📚 Libraries available:"
echo "   - secp256k1 (with recovery module)"
echo "   - OpenSSL"
echo "   - CUDA Toolkit"
echo "   - Google Test"

# تأیید نصب
echo "🔍 Verifying installations..."
pkg-config --exists libsecp256k1 && echo "✅ secp256k1: OK" || echo "❌ secp256k1: Failed"
pkg-config --exists openssl && echo "✅ OpenSSL: OK" || echo "❌ OpenSSL: Failed"
which nvcc && echo "✅ CUDA: OK" || echo "❌ CUDA: Failed"