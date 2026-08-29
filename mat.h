#include<iostream>
#include<stdio.h>
#include<vector>
#include<algorithm>
#include<string>
#include<ctype.h>
#include<random>
#include<numeric>
#include<fstream>
#include<iomanip>

struct Matrix {

    Matrix(): rows(0), cols(0), data(std::vector<float>()) {}

    Matrix(size_t r, size_t c):
        rows(r),
        cols(c),
        data(std::vector<float>(r * c))
    {}

    Matrix(size_t r, size_t c, std::string data_file):
     rows(r), cols(c), data(std::vector<float>()) {
        
        std::ifstream file(data_file, std::ios::binary | std::ios::ate);
        if(!file.is_open()) {
            std::cerr << "Failed to open data file to create matrix" << std::endl;
            return;
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        size_t elements = size / sizeof(float);
        if(elements != rows * cols) {
            std::cerr << "Matrix of specified dimensions cannot load the data" << std::endl;
            return;
        }
        data = std::vector<float>(elements);
        if (file.read((char*)(data.data()), size)) {
         std::cout << "Successfully loaded " << elements << " elements." << std::endl;
        }
        else {
            std::cerr << "Error reading file." << std::endl;
            return;
        }
    }

    ~Matrix() = default;

    void clear();
    void fill(float x);
    void scale(float scale);
    float sum() const;
    void display();

    size_t rows, cols;
    //row major storage
    std::vector<float> data;  
  
};

bool copyMat(Matrix* dst, const Matrix* src);
bool addMat(Matrix* out, const Matrix* a, const Matrix* b);
bool subMat(Matrix* out, const Matrix* a, const Matrix* b);
bool mulMat(Matrix* out, const Matrix* a, const Matrix* b,
     bool out_clear, bool aTranspose, bool bTranspose);
bool reluMat(Matrix* out, const Matrix* in);
bool softmaxMat(Matrix* out, const Matrix* in);
bool crossEntropyMat(Matrix* out, const Matrix* pred, const Matrix* y);

bool reluMatAddGradient(Matrix* out, const Matrix* in, const Matrix* grad);
bool softmaxMatAddGradient(Matrix* out, const Matrix* in, const Matrix* grad);
bool crossEntropyMatAddGradient(const Matrix* pred, const Matrix* y,
     Matrix* pred_grad, Matrix* y_grad, const Matrix* grad);

void Matrix::clear() {
    std::fill(data.begin(), data.end(), 0);
}

void Matrix::fill(float x) {
    std::fill(data.begin(), data.end(), x);
}

void Matrix::scale(float scale) {
    for(auto& a : data) {
        a *= scale;
    }
}

float Matrix::sum() const {
    return std::reduce(data.begin(), data.end(), 0.0F);
}

void Matrix::display() {
    for(auto i = 0; i < rows * cols; i++) {
        for(auto j = 0; j < cols; j++) {
            std::cout << std::setprecision(3) << data[i / cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

bool copyMat(Matrix* dst, const Matrix* src) {
    if(dst->rows != src->rows || dst->cols != src->cols) {
        return false;  
    }
    dst->data = src->data;
    return true;
}

bool addMat(Matrix* out, const Matrix* a, const Matrix* b) {
    if(a->rows != b->rows || a->cols != b->cols) {
        return false;
    }
    if(out->rows != a->rows || out->cols != a->cols) {
        return false;
    }
    size_t num_ele = out->cols * out->rows;
    for(auto i = 0; i < num_ele; i++){
        out->data[i] = a->data[i] + b->data[i];
    }
    return true;
}

bool subMat(Matrix* out, const Matrix* a, const Matrix* b) {
    if(a->rows != b->rows || a->cols != b->cols) {
        return false;
    }
    if(out->rows != a->rows || out->cols != out->rows) {
        return false;
    }
    size_t num_ele = out->cols * out->rows;
    for(auto i = 0; i < num_ele; i++){
        out->data[i] = a->data[i] - b->data[i];
    }
    return true;
}

//A @ B
bool mulMat(Matrix* out, const Matrix* a, const Matrix* b,
     bool out_clear, bool aTranspose, bool bTranspose) {
    size_t a_row = aTranspose ? a->cols : a->rows;
    size_t a_col = aTranspose ? a->rows : a->cols;
    size_t b_row = bTranspose ? b->cols : b->rows;
    size_t b_col = bTranspose ? b->rows : b->cols;
    
    if(a_col != b_row){
        return false;
    }
    if(out->rows != a_row || out->cols != b_col) {
        return false;
    }
    if(out_clear){
        out->clear();
    }
    //row major
    size_t m = a_row;
    size_t n = b_col;
    size_t k = a_col;
    for(auto i = 0; i < m * n; i++) {
        size_t row = i / n;
        size_t col = i % n;
        float dot = 0.0f;
        for(auto j = 0; j < k; j++){

            auto a_ele = aTranspose ? 
            a->data[j * a->cols + row] :
            a->data[row * a->cols + j];
            
            auto b_ele = bTranspose ? 
            b->data[col * b->cols + j] :
            b->data[j * b->cols + col];

            dot += a_ele * b_ele; 
        }
        out->data[i] += dot;
    }
    return true;
}

bool reluMat(Matrix* out, const Matrix* in) {
    if(out->rows != in->rows || out->cols != in->cols) {
        return false;
    }
    for(auto i = 0; i < in->data.size(); i++) {
        out->data[i] = std::max(0.0f, in->data[i]);
    }
    return true;
}

bool softmaxMat(Matrix* out, const Matrix* in) {
    if(out->rows != in->rows || out->cols != in->cols) {
        return false;
    }
    float expsum = 0.0f;
    for(auto i = 0; i < in->data.size(); i++) {
        out->data[i] = expf(in->data[i]);
        expsum += out->data[i];
    }
    out->scale(1.0f / expsum);
    return true;
}

bool crossEntropyMat(Matrix* out, const Matrix* pred, const Matrix* y) {
    if (pred->rows != y->rows || pred->cols != pred->cols) {
        return false;
    }
    if(out->rows != y->rows || out->cols != y->cols) {
        return false;
    }
    for(auto i = 0; i < out->data.size(); i++) {
        out->data[i] = (-1.0f) * y->data[i] * logf(pred->data[i]);
    }
    return true;
}

bool reluMatAddGradient(Matrix* out, const Matrix* in, const Matrix* grad) {
    if(out->rows != in->rows || out->cols != in->cols) {
        return false;
    }
    if(out->rows != grad->rows || out->cols != grad->cols) {
        return false;
    }    
    for(auto i = 0; i < in->data.size(); i++) {
        out->data[i] += in->data[i] > 0.0f ? grad->data[i] : 0.0f;
    }
    return true;    
}

bool crossEntropyMatAddGradient(const Matrix* pred, const Matrix* y,
     Matrix* pred_grad, Matrix* y_grad, const Matrix* grad) {
    if (pred->rows != y->rows || pred->cols != pred->cols) {
        return false;
    }

    if(y_grad) {
        if (y->rows != y_grad->rows || y->cols != y_grad->cols) {
            return false;
        }

        for(auto i = 0; i < y_grad->data.size(); i++) {
            y_grad->data[i] += -logf(pred->data[i]) * grad->data[i];
        }
    }

    if(pred_grad) {
        if (pred->rows != pred_grad->rows || pred->cols != pred_grad->cols) {
            return false;
        }

        for(auto i = 0; i < pred_grad->data.size(); i++) {
            pred_grad->data[i] += grad->data[i] * (-1.0f) * y->data[i] / pred->data[i];
        }
    }
}

bool softmaxMatAddGradient(Matrix* out, const Matrix* in, const Matrix* grad) {
    if(in->rows == 1 && in->cols == 1) {
        return false;
    }
    size_t jacobian_size = std::max(in->rows, in->cols);
    Matrix jacobian(jacobian_size, jacobian_size);
    
    for(auto i = 0; i < jacobian_size; i++) {
        for (auto j = 0; j < jacobian_size; j++) {
            jacobian.data[i * jacobian_size + j] =
            in->data[i] * ((i == j) - in->data[j]);
        }
    }
    auto res = mulMat(out, &jacobian, grad, 0, 0, 0);
    return true;
}