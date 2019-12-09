nvcc -lineinfo -rdc=true -std=c++14 part2.cu sha256.cu -arch=compute_35 -o part2 --compiler-bindir /usr/local/pub/swm/gcc-7.3.0/bin/g++ &&
./part2 0aca36d7d8e3bd46e6bab5bf3a47230e91e100ccd241c169e9d375f5b2a28f82 0000092a6893b712892a41e8438e3ff2242a68747105de0395826f60b38d88dc
