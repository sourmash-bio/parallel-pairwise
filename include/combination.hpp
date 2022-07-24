#include <vector>
#include <cstdint>


using namespace std;

class Combo {

public:
    Combo() = default;

    vector<std::pair<uint32_t, uint32_t>> combs;

    void combinations(int n) {
        this->combs.clear();
        this->comb(n, this->r, this->arr);
    }

private:
    int* arr = new int[2];
    int r = 2;

    void comb(int n, int r, int* arr) {
        for (int i = n; i >= r; i--) {
            arr[r - 1] = i;
            if (r > 1) comb(i - 1, r - 1, arr);
            else this->combs.emplace_back(std::make_pair(arr[0] - 1, arr[1] - 1));
        }
    }

};