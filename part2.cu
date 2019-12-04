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
// Find a nonce 
__global__ void find(unsigned char *thash, unsigned char *bhash, uint32_t *nonces, uint64_t *hashes)
{
    // Compute starting index and stride size
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = blockDim.x * gridDim.x;
    
    unsigned char *hhash2;//[64];
    unsigned char hhash[64+8];
    memcpy(hhash, bhash, 64);
    
    for(uint32_t ns = index; ns < INT32_MAX ; ns += stride)
    {
        // Once a satisfactory nonce has been found, be done
        if(pos > 0) { return; }

        // Counter to the example, the nonce is transformed into a 32-bit hex string (8 hex chars)
        unsigned char nc[8] = { h[(ns>>28)%16], h[(ns>>24)%16], h[(ns>>20)%16],
                                h[(ns>>16)%16], h[(ns>>12)%16], h[(ns>> 8)%16],
                                h[(ns>> 4)%16], h[ ns     %16] };
        memcpy(hhash+64, nc, 8);
        hhash2 = sha256(hhash, 72);
        hhash2 = sha256(hhash2, 64);

        // A nonce is satisfactory if it results in a hash less than the target hash
        if(less64(hhash2, thash) || pos < 5)
        {
            unsigned int mypos = atomicInc(&pos, 4);
            nonces[mypos] = ns;
            //hashes[mypos] = hhash2;
            //memcpy(hashes[mypos], hhash2, 64);
            free(hhash2);
        }
        else
        {
            free(hhash2);
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
    cudaMallocManaged(&bhg,    64*sizeof(char));
    cudaMallocManaged(&thg,    64*sizeof(char));
    cudaMallocManaged(&nonces,  4*sizeof(uint32_t)); // Let's start with max 4 nonces...
    cudaMallocManaged(&hashes,  4*sizeof(uint64_t)); // Let's start with max 4 nonces...

    // Determine structure - the number of blocks is rounded up
    int block_size = 256;
    int num_blocks = (UINT32_MAX + block_size - 1) / block_size;
    
    // Run kernal
    find<<<num_blocks, block_size>>>(thg, bhg, nonces, hashes);

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

    // Free memory
    cudaFree(bhg);
    cudaFree(thg);
    cudaFree(nonces);
    cudaFree(hashes);

    // Done
    return 0;
}
