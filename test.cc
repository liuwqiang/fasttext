#include <iostream>

#include "fasttext.h"

using namespace fasttext;

int main(int argc, char** argv) {

    FastText fasttext;
    fasttext.loadModel("/model/model.bin");
    Vector vec(128);
    std::vector<int32_t> line, labels;
    fasttext.textVector("农产品",vec,line,labels);
    std::cout << vec << std::endl;
}