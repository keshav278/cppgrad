#include "mat.h"

enum NodeFlags {
    NONE = 0,

    REQUIRES_GRAD     = (1 << 0),
    PARAMETER         = (1 << 1),
    INPUT             = (1 << 2),
    OUTPUT            = (1 << 3),
    OUTPUT_LABEL  = (1 << 4),
    LOSS_FN           = (1 << 5)

};

enum Op {
    NONE = 0,
    CREATE,

    UNARY_START,
    RELU,
    SOFTMAX,

    BINARY_START,
    ADD,
    SUB,
    MATMUL,
    CROSS_ENTROPY,    
};

inline size_t OpNumInputs(Op op) {
    if(op < Op::UNARY_START)
        return 0;
    if(op < Op::BINARY_START)
        return 1;
    return 2;
}

struct Node {
    size_t index{};
    NodeFlags flags{};
    Matrix value{};
    Matrix grad{};
    Op operation{};
    std::vector<Node*> inputs = std::vector<Node*>();
};

struct TopoGraph {
    size_t size{};
    std::vector<Node*> nodes = std::vector<Node*>();
};

struct ModelContext {
      size_t num_nodes{};
      
      Node input{};
      Node output{};
      Node label{};
      Node loss{};

      TopoGraph forward_pass{};
      TopoGraph loss_fn{};
};

Node createNodeForCtx(ModelContext* ctx, size_t rows, size_t cols,
                NodeFlags flags) {
    Node out;
    out.index = ctx->num_nodes++;
    out.flags = flags;
    out.value = Matrix(rows,cols);
    out.operation = Op::CREATE;
    if(flags & NodeFlags::REQUIRES_GRAD) {
        out.grad = Matrix(rows,cols);
    }
    if(flags & NodeFlags::INPUT) ctx->input = out;
    if(flags & NodeFlags::OUTPUT) ctx->output = out;
    if(flags & NodeFlags::OUTPUT_LABEL) ctx->label = out;
    if(flags & NodeFlags::LOSS_FN) ctx->loss = out;
    return out;
}

Node createOpNode(ModelContext* ctx, size_t rows, size_t cols,
     NodeFlags flags, Op op, Node* a, Node* b = nullptr) {

    if(a->flags & NodeFlags::REQUIRES_GRAD) {
        flags = (NodeFlags)(flags | NodeFlags::REQUIRES_GRAD);
    }
    if(b != nullptr && b->flags & NodeFlags::REQUIRES_GRAD) {
        flags = (NodeFlags)(flags | NodeFlags::REQUIRES_GRAD);
    }   
    Node out = createNodeForCtx(ctx, rows, cols, flags);
    
    out.inputs.push_back(a);
    if(b != nullptr) {
        out.inputs.push_back(b);
    }
    out.operation = op;
    return out;
}

TopoGraph graphCreate(ModelContext* ctx, Node* out_node) {
    //topo sort of the computational graph
    std::vector<bool> visited(ctx->num_nodes, false);
    std::vector<Node*> stack;
    std::vector<Node*> toposort;
    stack.push_back(out_node);
    //dfs
    while(stack.size() > 0){
        auto cur = stack.back(); 
        if(cur->index >= ctx->num_nodes) continue;
        //base
        if(visited[cur->index]){
            toposort.push_back(cur);
            stack.pop_back();
            continue;
        }
        //visit
        visited[cur->index] = true;

        //recurse
        for(auto node: cur->inputs) {
            stack.push_back(node);
        }
    }
    TopoGraph out;
    out.size = toposort.size();
    out.nodes = toposort;
    return out;

}

ModelContext* modelContextCreate() {

}

void graphCompile(ModelContext* ctx) {

}

void graphFeedForward(ModelContext* ctx) {

}

void modelContextCompute(ModelContext* ctx) {

}

void modelContextComputeGradients(ModelContext* ctx) {

}


