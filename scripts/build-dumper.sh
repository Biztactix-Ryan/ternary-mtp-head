#!/bin/bash
# rebuild llama-dump-nextn inside the ttf container
set -e
E=~/ttf/ttf-home/engine-q38
mkdir -p $E/examples/dump-nextn
cp ~/ttf/mtp/dump-nextn.cpp $E/examples/dump-nextn/dump-nextn.cpp
cp ~/ttf/mtp/CMakeLists.txt $E/examples/dump-nextn/CMakeLists.txt
grep -q dump-nextn $E/examples/CMakeLists.txt || \
  sed -i 's/    add_subdirectory(embedding)/    add_subdirectory(embedding)\n    add_subdirectory(dump-nextn)/' $E/examples/CMakeLists.txt
cd $E
cmake -B build-cuda . > /dev/null
cmake --build build-cuda --target llama-dump-nextn -j 32 2>&1 | tail -15
ls -la build-cuda/bin/llama-dump-nextn
