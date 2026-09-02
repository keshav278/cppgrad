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

//    Relu Relu Softmax
// 784 -> 16 -> 16 -> 10 MLP
//        |_____| residual connection
//     W0b0  W1b1  W2b2
void createMnistModel(ModelContext* ctx) {
    Node* input = createNodeForCtx(ctx, 784, 1, NodeFlags::INPUT);

    //graph for model
    NodeFlags parameter = (NodeFlags)(NodeFlags::PARAMETER | NodeFlags::REQUIRES_GRAD);
    Node* w0 = createNodeForCtx(ctx, 16, 784, parameter);
    Node* w1 = createNodeForCtx(ctx, 16, 16, parameter);
    Node* w2 = createNodeForCtx(ctx, 10, 16, parameter);

    //kaiming init for relu activations
    float bound0 = sqrtf(6.0f / (784));
    float bound1 = sqrtf(6.0f/ (16));
    w0->value.fill_rand_uniform(-bound0, bound0);
    w1->value.fill_rand_uniform(-bound1, bound1);
    w2->value.fill_rand_uniform(-bound1, bound1);

    Node* b0 = createNodeForCtx(ctx, 16, 1, parameter);
    Node* b1 = createNodeForCtx(ctx, 16, 1, parameter);
    Node* b2 = createNodeForCtx(ctx, 10, 1, parameter);

    Node* z0_a = createOpNode(ctx, w0->value.rows, input->value.cols,
         NodeFlags::NO_FLAG, Op::MATMUL, w0, input);
    Node* z0 = createOpNode(ctx, z0_a->value.rows, z0_a->value.cols,
         NodeFlags::NO_FLAG, Op::ADD, z0_a, b0);
    Node* a0 = createOpNode(ctx, z0->value.rows, z0->value.cols,
         NodeFlags::NO_FLAG, Op::RELU, z0);

    Node* z1_a = createOpNode(ctx, w1->value.rows, a0->value.cols,
         NodeFlags::NO_FLAG, Op::MATMUL, w1, a0);
    Node* z1 = createOpNode(ctx, z1_a->value.rows, z1_a->value.cols,
         NodeFlags::NO_FLAG, Op::ADD, z1_a, b1);
    Node* a1_a = createOpNode(ctx, z1->value.rows, z1->value.cols,
         NodeFlags::NO_FLAG, Op::RELU, z1);
    Node* a1 = createOpNode(ctx, a1_a->value.rows, a1_a->value.cols,
         NodeFlags::NO_FLAG, Op::ADD, a1_a, a0); 
         
    Node* z2_a = createOpNode(ctx, w2->value.rows, a1->value.cols,
         NodeFlags::NO_FLAG, Op::MATMUL, w2, a1);
    Node* z2 = createOpNode(ctx, z2_a->value.rows, z2_a->value.cols,
         NodeFlags::NO_FLAG, Op::ADD, z2_a, b2);

    Node* output = createOpNode(ctx, z2->value.rows, z2->value.cols,
         NodeFlags::OUTPUT, Op::SOFTMAX, z2);

    Node* label = createNodeForCtx(ctx, 10, 1, NodeFlags::OUTPUT_LABEL);

    Node* loss = createOpNode(ctx, 1, 1,
         NodeFlags::LOSS_FN, Op::CROSS_ENTROPY, label, output);

    
}
void modelTrain(ModelContext* ctx, const TrainingParams* params);


int main()
{
    Matrix train_images = Matrix(60000, 784, "X_train.bin");
    Matrix test_images = Matrix(10000, 784, "X_test.bin");
    Matrix train_labels = Matrix(60000, 10, "y_train.bin");
    Matrix test_labels = Matrix(10000, 10, "y_test.bin");

    for (int i = 100; i < 125; i ++) {
        std::cout << train_labels.data[i] << " ";
    }
    auto ctx = std::make_unique<ModelContext>();
    std::cout << "\ncreating model..\n";
    createMnistModel(ctx.get());
    std::cout << "compiling model..\n";
    modelContextCompile(ctx.get());

    ctx.get()->input->value.data = std::vector<float>(train_images.data.begin(), train_images.data.begin() + 784);
    std::cout << "doing a forward pass..\n";
    modelContextFeedForward(ctx.get());

    std::cout << "\n\npretraining output\n";
    for (int i = 0; i < 10; i++) {
        std::cout << ctx.get()->output->value.data[i] << " ";
    }
}