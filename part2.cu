#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>

#include "sha256.cuh"

// Kernal code
__device__ unsigned int pos = 0; // position in nonce result array
// Check two char strings for equality
__device__ bool less64(unsigned char *lhs, unsigned char *rhs)
{
    for(unsigned i = 0; i < 64; ++i)
    {
        if(lhs[i] == rhs[i])     { continue; }
        else if(lhs[i] < rhs[i]) { return true; }
        else                     { return false;}
    }
    // Two identical strings are not less than each other
    return false;
}
// Kernal: find a nonce 
__global__ void find(unsigned char *thash, unsigned char *bhash, uint32_t *nonces, uint64_t *hashes, unsigned int *gpos)
{
//    *gpos = 999;
    // Compute starting index and stride size
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = blockDim.x * gridDim.x;
//    *gpos = index % UINT_MAX;
    
    unsigned char *hhash2;//[64];
    unsigned char hhash[64+8];
    memcpy(hhash, bhash, 64);
    
    for(uint32_t ns = index; ns < INT32_MAX ; ns += stride)
    {
//        *gpos = ns / 256;
        // Once a satisfactory nonce has been found, be done
        if(pos > 0) { return; }

        // Counter to the example, the nonce is transformed into a 32-bit hex string (8 hex chars)
        unsigned char nc[8] = { h[(ns>>28)%16], h[(ns>>24)%16], h[(ns>>20)%16],
                                h[(ns>>16)%16], h[(ns>>12)%16], h[(ns>> 8)%16],
                                h[(ns>> 4)%16], h[ ns     %16] };
        memcpy(hhash+64, nc, 8);
//        for(unsigned i = 0; i < 8; ++i) { hhash[64+i] = nc[i]; }
        hhash2 = sha256(hhash, 72);
        hhash2 = sha256(hhash2, 64);

//        hhash2 = (unsigned char *) malloc(64);
////        *hhash2 = "0000008888888882892a41e8438e3ff2242a68747105de0395826f60b38d88dc";
//        memset(hhash2, '0', 64);
////        std::string ts = std::to_string(ns);
////        for(int i = 0; i < 32 && i < 32; ++i) { hhash2[63-i] = h[ns >> i % 16]; }
//        for(unsigned i = 0; i < 8; ++i) { hhash2[63-i] = nc[i]; }

        // A nonce is satisfactory if it results in a hash less than the target hash
        if(less64(hhash2, thash))// || pos < 3)
        {
            unsigned int mypos = atomicInc(&pos, 4);
            nonces[mypos] = ns;
            //memcpy(&(nonces[mypos]), nc, 8);
//            nonces[mypos] = ((uint64_t)nc[0] << 56) + ((uint64_t)nc[1] << 48) + ((uint64_t)nc[2] << 40) +
//                            ((uint64_t)nc[3] << 32) + ((uint64_t)nc[4] << 24) + (nc[5] << 16) +
//                            (nc[6] <<  8) + nc[7];
            //hashes[mypos] = hhash2;
            memcpy(&(hashes[mypos]), hhash2, 64);
            free(hhash2);
            *gpos = mypos+200; // just for testing
//            return;
        }
        else
        {
            free(hhash2);
//            if(pos > 3) { return; }
        }
    }
}

int main(int argc, char **argv)
{
    // Handle too few arguments
    if(argc < 3)
    {
        std::cout << "Too few arguments." << std::endl;
        return 1;
    }
    
    // Read inputs
    std::string bhashstr = argv[1];
    std::string thashstr = argv[2];


    // Allocate Unified Memory
    unsigned char *bhg, *thg;
    uint32_t *nonces;
    uint64_t *hashes;
    unsigned int *gpos;     // Purely for testing to see if we get output from the kernal
    cudaMallocManaged(&bhg,    64*sizeof(char));
    cudaMallocManaged(&thg,    64*sizeof(char));
    cudaMallocManaged(&nonces,  4*sizeof(uint32_t)); // Let's start with max 4 nonces...
    cudaMallocManaged(&hashes,  4*sizeof(uint64_t)); // Let's start with max 4 nonces...
    cudaMallocManaged(&gpos,      sizeof(unsigned int));
    //*gpos = 100;

    // Determine structure - the number of blocks is rounded up
    int block_size = 256;
    int num_blocks = UINT32_MAX / block_size;
//    int num_blocks = 256;
//    int block_size = UINT32_MAX / num_blocks;
    
    // Run kernal
    find<<<num_blocks, block_size>>>(thg, bhg, nonces, hashes, gpos);

    std::cout <<   "BlockHash: " << bhashstr
              << "\nTargetHash: " << thashstr
              << "\nKernal launched, waiting for kernal..." << std::endl;

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Report to user
    for(int i = 0; i < 16 ; ++i)
    {
        if(nonces[i] == 0) { continue; }
        std::cout << "Resulting Hash: " << hashes[i] << std::endl;
        std::cout << "Nonce:" << (int32_t) nonces[i] << std::endl;
    }
    std::cout << *gpos << std::endl;

    // Free memory
    cudaFree(bhg);
    cudaFree(thg);
    cudaFree(nonces);
    cudaFree(hashes);
    cudaFree(gpos);

    // Done
    return 0;
}
