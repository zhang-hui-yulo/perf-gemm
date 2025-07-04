#include "flash_attention_2_sm80.hpp"

int main(int argc, char** argv) {
    using namespace cute;
    auto tA = make_tensor<int32_t>(make_shape(_128{}, _64{}), make_stride(_64{}, _1{}));
}