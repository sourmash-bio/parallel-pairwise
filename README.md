# parallel-pairwise

Attempt to brute-force sourmash signatures pairwise comparisons. Result is written in a csv files.

## Build
```bash
cmake -Bbuild -DCMAKE_BUILD_TYPE=RELEASE
cmake --build build -j 25
# ./build/pairwise <sigs_directory> <threads> <output>"
```

## Clustering

First: install dependecies
```bash
pip install -r requirements.txt
```

Second: run clustering
```bash
python cluster_it.py --csv CSV --cutoff CUTOFF --output OUTPUT
```


## Example:
./build/pairwise test_data 16 pairwise_test_data
python cluster_it.py --csv pairwise_test_data.csv --cutoff 10 --output clusters_test_data.txt
