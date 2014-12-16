a.out: average.cc 
#	g++ -DNDEBUG -std=c++11 -O3 -m64 -mcx16 -flto -fabi-version=0 -fdata-sections -ffunction-sections -I. -Wall -Wno-attributes -Wno-invalid-offsetof -Wl,--gc-sections -march=native average.cc -lrt -lpthread
	g++ -std=c++11 -O3 -m64 -flto -mavx2 -fabi-version=0 -fdata-sections -ffunction-sections -I. -Wall -Wno-attributes -Wno-invalid-offsetof -Wl,--gc-sections -march=native average.cc -lrt -lpthread

clean:
	rm -f a.out a.out.s

all: clean a.out
	objdump -CDzhxs --insn-width=10 ./a.out > a.out.s
