ffbpn: main.o ffbpn.o
	gcc -o ffbpn main.o ffbpn.o

main.o: main.c ffbpn.c
	gcc -Wall -std=c11 -c main.c ffbpn.c

ffbpn.o: ffbpn.c ffbpn.h
	gcc -Wall -std=c11 -c ffbpn.c ffbpn.h

clear: *.o
	rm *.o