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

    Node* loss = createOpNode(ctx, 10, 1,
         NodeFlags::LOSS_FN, Op::CROSS_ENTROPY, output, label);

    
}
void modelTrain(ModelContext* ctx, const TrainingParams* params) {
     Matrix* train_images = params->train_images;
     Matrix* train_labels = params->train_labels;
     Matrix* test_images = params->test_images;
     Matrix* test_labels = params->test_labels;

     size_t num_examples = train_images->rows;
     size_t input_size = train_images->cols;
     size_t output_size = train_labels->cols;
     size_t num_tests = test_images->rows;

     size_t num_batches = num_examples / params->batch_size;

     std::vector<size_t> indices(num_examples);
     std::iota(indices.begin(), indices.end(), 0);
     std::random_device rd;
     std::mt19937 g(rd());
     
     //stochastic gradient descent
     for(auto epoch = 0; epoch < params->epochs; epoch++){
          //shuffle access to the examples
          std::shuffle(indices.begin(), indices.end(), g);

          

          for(auto batch = 0; batch < num_batches; batch++) {
               //clear gradients for all parameters
               for(auto i = 0; i < ctx->loss_fn.size; i++) {
                    Node* cur = ctx->loss_fn.nodes[i];
                    if(cur->flags & NodeFlags::PARAMETER) {
                         cur->grad.clear();
                    }
               }
               float avg_cost = 0.0f;

               for(auto i = 0; i < params->batch_size; i++) {
                    auto access_index = batch * params->batch_size + i;
                    auto example_index = indices[access_index];

                    ctx->input->value.data = 
                    std::vector<float>(train_images->data.begin() + example_index * input_size,
                     train_images->data.begin() + (example_index + 1) * input_size);

                    ctx->label->value.data = 
                    std::vector<float>(train_labels->data.begin() + example_index * output_size,
                     train_labels->data.begin() + (example_index + 1) * output_size); 

                    graphCompute(&ctx->loss_fn);

                    graphComputeGradients(&ctx->loss_fn);

                    avg_cost += ctx->loss->value.sum();
               }
               avg_cost /= (float)params->batch_size;

               for(auto i = 0; i < ctx->loss_fn.size; i++) {
                    Node* cur = ctx->loss_fn.nodes[i];

                    if(cur->flags & NodeFlags::PARAMETER) {
                         cur->grad.scale(params->learning_rate / params->batch_size);
                         subMat(&cur->value, &cur->value, &cur->grad);
                    }
               }

               std::printf("Epoch: %d / %d, Batch: %d / %d, Avg Cost: %.4f\r",
               epoch + 1, params->epochs, batch + 1, num_batches, avg_cost
               );
          }
          std::cout << "\n";
     }

     //evaluating on test data
     size_t correct = 0;
     float avg_cost = 0.0f;

     for(auto i = 0; i < num_tests; i++) {
          ctx->input->value.data = 
          std::vector<float>(test_images->data.begin() + i * input_size,
               test_images->data.begin() + (i + 1) * input_size);

          ctx->label->value.data = 
          std::vector<float>(test_labels->data.begin() + i * output_size,
               test_labels->data.begin() + (i + 1) * output_size); 

          graphCompute(&ctx->loss_fn);

          avg_cost += ctx->loss->value.sum();
          correct += ctx->label->value.argmax() == ctx->output->value.argmax();
     }
     avg_cost /= (float)num_tests;

     std::cout<<"\nTest Completed!\n";
     printf("Accuracy: %d / %d (%.2f%%), Average Cost: %.4f\n",
     correct, num_tests, (float) correct / num_tests * 100.0f, avg_cost);
}

void draw_mnist_digit(float* data) {
    for (uint32_t y = 0; y < 28; y++) {
        for (uint32_t x = 0; x < 28; x++) {
            float num = data[x + y * 28];
            uint32_t col = 232 + (uint32_t)(num * 23);
            printf("\x1b[48;5;%dm  ", col);
        }
        printf("\n");
    }
    printf("\x1b[0m");
}


int main()
{
    Matrix train_images = Matrix(60000, 784, "X_train.bin");
    Matrix test_images = Matrix(10000, 784, "X_test.bin");
    Matrix train_labels = Matrix(60000, 10, "y_train.bin");
    Matrix test_labels = Matrix(10000, 10, "y_test.bin");

     if(train_images.data.size() != 60000 * 784 ||
        train_labels.data.size() != 60000 * 10 ||
        test_images.data.size() != 10000 * 784 ||
        test_labels.data.size() != 10000 * 10) {
          std::cerr << "MNIST data files are missing or have unexpected sizes.\n";
          return 1;
     }

    draw_mnist_digit(test_images.data.data() + 784);
    for (int i = 10; i < 20; i++) {
        std::cout << test_labels.data[i] << " ";
    }
    auto ctx = std::make_unique<ModelContext>();
    std::cout << "\ncreating model..\n";
    createMnistModel(ctx.get());
    std::cout << "compiling model..\n";
    modelContextCompile(ctx.get());

    ctx.get()->input->value.data = std::vector<float>(test_images.data.begin() + 784 * 10, test_images.data.begin() + 784 * 11);
    std::cout << "doing a forward pass..\n";
    modelContextFeedForward(ctx.get());

    std::cout << "\n\npretraining output\n";
    for (int i = 10; i < 20; i++) {
        std::cout << std::setprecision(4) << ctx.get()->output->value.data[i] << " ";
    }

    TrainingParams params{
         &train_images,
         &train_labels,
         &test_images,
         &test_labels,
         3,
         50,
         0.01f
    };
    std::cout << "\n\nTraining model...\n\n";
    modelTrain(ctx.get(), &params);

    std::cout << "\n\n posttraining output\n";
    for (int i = 10; i < 20; i++) {
        std::cout << std::setprecision(4) << ctx.get()->output->value.data[i] << " ";
    }
}