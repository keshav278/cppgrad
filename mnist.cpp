#include "graph.h"

struct TrainingParams {
    Matrix* train_images;
    Matrix* train_labels;
    Matrix* test_images;
    Matrix* test_labels;

    size_t epochs;
    size_t batch_size;
    float learning_rate;

};

void modelTrain(ModelContext* ctx, const TrainingParams* params);


int main()
{
    Matrix train_images = Matrix(60000, 784, "X_train.bin");
    Matrix test_images = Matrix(10000, 784, "X_test.bin");
    Matrix train_labels = Matrix(60000, 10, "y_train.bin");
    Matrix test_labels = Matrix(10000, 10, "y_test.bin");

    for (int i = 0; i < 10; i ++) {
        std::cout << train_labels.data[i] << " ";
    }
}