# parallel-pairwise

Attempt to brute-force sourmash signatures pairwise comparisons. Result in written in a csv files.

## Build
```bash
cmake -Bbuild -DCMAKE_BUILD_TYPE=RELEASE
cmake --build build -j 25
# ./build/pairwise <sigs_directory> <threads> <output>"
./build/pairwise test_sigs 16 pairwise_test_sigs
```
