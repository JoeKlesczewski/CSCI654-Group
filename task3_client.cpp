#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

#define portno 1234

void error(const char *msg)
{
    perror(msg);
    exit(0);
}

int main(int argc, char *argv[])
{
    int sockfd, n;
    struct sockaddr_in serv_addr;
    struct hostent *server;
    char *confirmation = "Got message from server.\n";
    char *bhashstr;
    char *thashstr;

    char buffer[256];

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
        error("ERROR opening socket");
    server = gethostbyname(argv[1]);
    if (server == NULL) {
        fprintf(stderr,"ERROR, no such host\n");
        exit(0);
    }
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr,
         (char *)&serv_addr.sin_addr.s_addr,
         server->h_length);
    serv_addr.sin_port = htons(portno);
    if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0)
        error("ERROR connecting");
    printf("Please enter the message: ");
    bzero(buffer,256);


    // get bhashstr
    bzero(buffer,256);
    n = read(sockfd,buffer,255);
    if (n < 0) error("ERROR reading bhashstr from server");
    printf("%s\n",buffer);
    n = write(sockfd,confirmation,strlen(confirmation));
    if (n < 0) error("ERROR writing bhashstr confirmation to socket");
    bhashstr = buffer;
    printf("Bhashstr is: %s\n", bhashstr);

    // get thashstr
    bzero(buffer,256);
    n = read(sockfd,buffer,255);
    if (n < 0) error("ERROR reading thashstr from server");
    printf("%s\n",buffer);
    n = write(sockfd,confirmation,strlen(confirmation));
    if (n < 0) error("ERROR writing thashstr confirmation to socket");
    thashstr = buffer;
    printf("Thashstr is: %s\n", thashstr);


    close(sockfd);
    return 0;
}
