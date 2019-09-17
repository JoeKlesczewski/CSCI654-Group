#include <cstdint>
#include <iostream>
#include <omp.h>
#include <string>

#include "sha256.h"

#define NUM_THREADS 8

int main(int argc, char **argv)
{
    if(argc < 3)
    {
        std::cout << "ERROR: Not enough arguments." << std::endl;
        return 2;
    }

    // Read inputs
    std::string bhashstr = argv[1];
    std::string thashstr = argv[2];
    // User output
    std::cout << "BlockHash: " << bhashstr << std::endl;
    std::cout << "TargetHash: "<< thashstr << std::endl;
    std::cout << "Performing Proof-of_work...wait..." << std::endl;
    // Set up loop variables
    int32_t nonce = INT32_MIN;
    std::string hashstr;
    // Find a nonce that gives a hash less than the target hash
    #pragma omp parallel num_threads(NUM_THREADS) private(nonce, hashstr)
    {
        std::string tmp_hashstr;
        int id = omp_get_thread_num();
        nonce += (INT32_MAX / NUM_THREADS) * 2 * id;
        for(; nonce < INT32_MAX; ++nonce)
        {
            // Per the example, the nonce is stringified as a decimal string including leading -
            tmp_hashstr = sha256(sha256(bhashstr + std::to_string(nonce)));
            if(tmp_hashstr < thashstr) {  }
        }
    }
    // Report results
    std::cout << "Resulting Hash: " << hashstr << std::endl;
    std::cout << "Nonce:" << nonce << std::endl;
}
