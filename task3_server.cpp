#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#define portno 1234

void dostuff(int, char*, char*, int); /* function prototype */
void error(const char *msg)
{
    perror(msg);
    exit(1);
}

int main(int argc, char *argv[])
{
    char *bhashstr = "0aca36d7d8e3bd46e6bab5bf3a47230e91e100ccd241c169e9d375f5b2a28f82";
    char *thashstr = "0000092a6893b712892a41e8438e3ff2242a68747105de0395826f60b38d88dc";
    int sockfd, newsockfd, pid;
    socklen_t clilen;
    struct sockaddr_in serv_addr, cli_addr;
    int counter = 0;

     sockfd = socket(AF_INET, SOCK_STREAM, 0);
     if (sockfd < 0)
        error("ERROR opening socket");
     bzero((char *) &serv_addr, sizeof(serv_addr));
     serv_addr.sin_family = AF_INET;
     serv_addr.sin_addr.s_addr = INADDR_ANY;
     serv_addr.sin_port = htons(portno);
     if (bind(sockfd, (struct sockaddr *) &serv_addr,
              sizeof(serv_addr)) < 0)
              error("ERROR on binding");
     listen(sockfd,5);
     clilen = sizeof(cli_addr);
     while (1) {
         newsockfd = accept(sockfd,
               (struct sockaddr *) &cli_addr, &clilen);
         if (newsockfd < 0)
             error("ERROR on accept");
         pid = fork();
         if (pid < 0)
             error("ERROR on fork");
         if (pid == 0)  {
             close(sockfd);
             dostuff(newsockfd, bhashstr, thashstr, counter);
             exit(0);
         }
         else close(newsockfd);
         counter = counter + 1;
     } /* end of while */
     close(sockfd);
     return 0; /* we never get here */
}

/******** DOSTUFF() *********************
 There is a separate instance of this function
 for each connection.  It handles all communication
 once a connnection has been established.
 *****************************************/
void dostuff (int sockfd, char* bhashstr, char* thashstr, int counter)
{
    int n;
    char buffer[256];
    printf("%d", counter);
    int32_t nonce = INT32_MIN;
    int32_t max_nonce;

    // send bhashstr to client
    n = write(sockfd,bhashstr,strlen(bhashstr));
    if (n < 0) error("ERROR writing bhashstr to client");
    bzero(buffer,256);
    n = read(sockfd,buffer,255);
    if (n < 0) error("ERROR reading bhashstr confirmation from client");
    printf("%s\n",buffer);

    // send thashstr to client
    n = write(sockfd,thashstr,strlen(thashstr));
    if (n < 0) error("ERROR writing thashstr to client");
    bzero(buffer,256);
    n = read(sockfd,buffer,255);
    if (n < 0) error("ERROR reading thashstr confirmation from client");
    printf("%s\n",buffer);



}
