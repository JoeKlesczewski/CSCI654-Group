#include <climits>
#include <iostream>
#include <string>


int main(int argc, char *argv)
{
    if(argc < 3)
    {
        std::cout << "ERROR: Not enough arguments." << std::endl;
    }

    long nonce = LONG_MIN;
    std::string bhashstr(argv[1]);
    std::string thashstr(argv[2]);
    
}
