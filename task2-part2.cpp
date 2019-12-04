#include <cstdint>
#include <iostream>
#include <omp.h>
#include <string>

#include "sha256.h"

#define OMP_CANCELLATION TRUE
#define NUM_THREADS 8

std::string HEXTIMESTWO(std::string in)
{
    unsigned long long result = strtoull(in.c_str(), NULL, 16);
    result *= 2;
    char buf[64];
    snprintf (buf, 64, "%lX", result);
    return std::string(buf);
}

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

    for(int i = 0; i < 10; ++i)
    {
        // User output
        std::cout << "BlockHash: " << bhashstr << std::endl;
        std::cout << "TargetHash: "<< thashstr << std::endl;
        std::cout << "Performing Proof-of_work...wait..." << std::endl;
        // Set up loop variables
        int32_t out_nonce;
        std::string out_hashstr;
        // Find a nonce that gives a hash less than the target hash
        #pragma omp parallel num_threads(NUM_THREADS)
        {
            std::string hashstr;
            int id = omp_get_thread_num();
            for(int32_t nonce = INT32_MIN + (INT32_MAX / NUM_THREADS) * 2 * id; nonce < INT32_MAX; ++nonce)
            {
                // Per the example, the nonce is stringified as a decimal string including leading -
                hashstr = sha256(sha256(bhashstr + std::to_string(nonce)));
                if(hashstr < thashstr)
                {   // Found the correct nonce - now need to return from this thread and terminate
                    out_nonce = nonce;
                    out_hashstr = hashstr;
                    #pragma omp cancel parallel
                }
                #pragma omp cancellation point parallel
            }
        }
        // Report results
        std::cout << "Resulting Hash: " << out_hashstr << std::endl;
        std::cout << "Nonce:" << out_nonce << std::endl;
        bhashstr = sha256(bhashstr + std::to_string(out_nonce));
        thashstr = HEXTIMESTWO(thashstr);
        std::cout << "New block hash:  " << bhashstr << std::endl;
        std::cout << "New target hash: " << thashstr << std::endl;
    }
}
