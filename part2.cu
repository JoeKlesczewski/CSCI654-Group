#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>

#include "sha256.cuh"

// Kernal code
__device__ unsigned char ddbhg[64];
__device__ unsigned char ddthg[64];
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
        if(less64(hhash2, thash))// || pos == 0)// || pos < 3)
        {
            unsigned int mypos = atomicInc(&pos, 4);
            //nonces[mypos] = ns;
            *nonces = ns;
            //memcpy(nonces, nc, 8);
            memcpy(hashes, hhash2, 64);
            //*hashes = hhash2;
            free(hhash2);
            return;
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
    unsigned char *dbhg, *dthg;
    uint32_t *dnonces;
    uint64_t *dhashes;
    unsigned char *bhg, *thg;
    bhg = (unsigned char *) malloc(64 * sizeof(unsigned char));
    thg = (unsigned char *) malloc(64 * sizeof(unsigned char));
    uint32_t nonces = 0;
    uint64_t hashes = 0;
//    uint32_t *nonces = (uint32_t) malloc(sizeof(uint32_t);
//    uint64_t *hashes = (uint64_t) malloc(sizeof(uint64_t);

/*    cudaMallocManaged(&bhg,    64*sizeof(char));
    cudaMallocManaged(&thg,    64*sizeof(char));
    cudaMallocManaged(&nonces,  4*sizeof(uint32_t)); // Let's start with max 4 nonces...
    cudaMallocManaged(&hashes,  4*sizeof(uint64_t)); // Let's start with max 4 nonces...
*/
    //bhg = bhashstr.c_str();
    for(int i = 0; i < bhashstr.size(); ++i)
    {
        bhg[i] = bhashstr.c_str()[i];
        thg[i] = thashstr.c_str()[i];
    }
    cudaMalloc((void **) &dbhg, 64*sizeof(char));
    cudaMalloc((void **) &dthg, 64*sizeof(char));
    cudaMemcpy(dbhg, &bhg, 64*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dthg, &thg, 64*sizeof(char), cudaMemcpyHostToDevice);

    cudaMalloc((void **) &dnonces, sizeof(uint32_t));
    cudaMalloc((void **) &dhashes, sizeof(uint64_t));
    cudaMemcpy(dnonces, &nonces, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dhashes, &hashes, sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Determine structure - the number of blocks is rounded up
    int block_size = 256;
    int num_blocks = UINT32_MAX / block_size;
//    int num_blocks = 256;
//    int block_size = UINT32_MAX / num_blocks;
    
    // Run kernal
    find<<<num_blocks, block_size>>>(dthg, dbhg, dnonces, dhashes);
    
    cudaMemcpy(&nonces, dnonces, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hashes, dhashes, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    std::cout <<   "BlockHash: " << bhashstr
              << "\nTargetHash: " << thashstr
              << "\nKernal launched, waiting for kernal..." << std::endl;

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Report to user
    std::cout << "resulting hash: " << hashes << std::endl;
    std::cout << "nonce:" << (int32_t) nonces << std::endl;
/*    for(int i = 0; i < 16 ; ++i)
    {
        if(nonces[i] == 0) { continue; }
        std::cout << "resulting hash: " << hashes[i] << std::endl;
        std::cout << "nonce:" << (int32_t) nonces[i] << std::endl;
    }
*/
    // Free memory
/*    cudaFree(bhg);
    cudaFree(thg);
    cudaFree(nonces);
    cudaFree(hashes);
    cudaFree(gpos);
*/
    // Done
    return 0;
}
