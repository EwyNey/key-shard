.PHONY: all clean build debug release reldeb

debug:
	rm -rf build && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Debug ..
release:
	rm -rf build && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=release ..
reldeb:
	rm -rf build && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
build:
	cd ./build && make
