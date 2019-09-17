#include <cstdint>
#include <iostream>
#include <string>

#include "sha256.h"


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
    for(nonce = INT32_MIN; nonce < INT32_MAX; ++nonce)
    {
        // Question: should to_string really be a decimal string?? It is in the example...
        hashstr = sha256(sha256(bhashstr + std::to_string(nonce)));
        if(hashstr < thashstr) { break; }
    }
    // Report results
    std::cout << "Resulting Hash: " << hashstr << std::endl;
    std::cout << "Nonce:" << nonce << std::endl;
}
