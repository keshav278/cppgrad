#include "mat.h"

enum NodeFlags {
    NO_FLAG = 0,

    REQUIRES_GRAD     = (1 << 0),
    PARAMETER         = (1 << 1),
    INPUT             = (1 << 2),
    OUTPUT            = (1 << 3),
    OUTPUT_LABEL      = (1 << 4),
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
    std::vector<Node*> inputs;
};

struct TopoGraph {
    size_t size{};
    std::vector<Node*> nodes;
};

struct ModelContext {
      size_t num_nodes{};

      std::vector<std::unique_ptr<Node>> nodes;

      Node* input{};
      Node* output{};
      Node* label{};
      Node* loss{};

      TopoGraph forward_pass{};
      TopoGraph loss_fn{};
};

 Node* createNodeForCtx(ModelContext* ctx, size_t rows, size_t cols,
                NodeFlags flags) {
    auto out = std::make_unique<Node>();
    out->index = ctx->num_nodes++;
    out->flags = flags;
    out->value = Matrix(rows,cols);
    out->operation = Op::CREATE;
    if(flags & NodeFlags::REQUIRES_GRAD) {
        out->grad = Matrix(rows,cols);
    }

    Node* out_obj = out.get();
    ctx->nodes.push_back(std::move(out));

    if(flags & NodeFlags::INPUT) (ctx->input) = out_obj;
    else if(flags & NodeFlags::OUTPUT) (ctx->output) = out_obj;
    else if(flags & NodeFlags::OUTPUT_LABEL) (ctx->label) = out_obj;
    else if(flags & NodeFlags::LOSS_FN) (ctx->loss) = out_obj;
    return out_obj;
}

Node* createOpNode(ModelContext* ctx, size_t rows, size_t cols,
     NodeFlags flags, Op op, Node* a, Node* b = nullptr) {

    if(a->flags & NodeFlags::REQUIRES_GRAD) {
        flags = (NodeFlags)(flags | NodeFlags::REQUIRES_GRAD);
    }
    if(b != nullptr && b->flags & NodeFlags::REQUIRES_GRAD) {
        flags = (NodeFlags)(flags | NodeFlags::REQUIRES_GRAD);
    }   
    Node* out = createNodeForCtx(ctx, rows, cols, flags);
    
    out->inputs.push_back(a);
    if(b != nullptr) {
        out->inputs.push_back(b);
    }
    out->operation = op;
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

void graphCompute(TopoGraph* graph) {
    
    for (auto node : graph->nodes) {   

        if (node->inputs.empty()) {
            continue;
        }

        Node* a = node->inputs[0];
        Node* b = nullptr;
        if(node->inputs.size() == 2){
            b = node->inputs[1];
        }
        bool res = true;
        switch(node->operation) {
            case Op::NONE:
            case Op::CREATE: break;

            case Op::UNARY_START: break;

            case Op::RELU: {
                res &= reluMat(&node->value, &a->value);
            } break;
            case Op::SOFTMAX: {
                res &= softmaxMat(&node->value, &a->value);
            } break;

            case Op::BINARY_START: break;

            case Op::ADD: {
                if(b == nullptr) {res = false; break;}
                res &= addMat(&node->value, &a->value, &b->value); 
            } break;
            case Op::SUB: {
                if(b == nullptr) {res = false; break;}
                res &= subMat(&node->value, &a->value, &b->value); 
            } break;
            case Op::MATMUL: {
                if(b == nullptr) {res = false; break;}
                res &= mulMat(&node->value, &a->value, &b->value, true,false,false); 
            } break;
            case Op::CROSS_ENTROPY: {
                if(b == nullptr) {res = false; break;}
                res &= crossEntropyMat(&node->value, &a->value, &b->value);
            } break;

        }
        if(!res)
            std::cout << "Error in graph compute\n";

    }
}

void graphComputeGradients(TopoGraph* graph) {
    for(auto cur : graph->nodes) {
        if(!(cur->flags & NodeFlags::REQUIRES_GRAD)){
            continue;
        }
        if(cur->flags & NodeFlags::PARAMETER){
            continue;
        }
        cur->grad.clear();
    }
    graph->nodes[graph->size - 1]->grad.fill(1.0f); // df/df = 1

    for(long int i = graph->size - 1; i >= 0; i--) {
        auto cur = graph->nodes[i];

        if(!(cur->flags & NodeFlags::REQUIRES_GRAD)) {
            continue;
        }

        if (cur->inputs.empty()) {
            continue;
        }

        auto a = cur->inputs[0];
        auto b = cur->inputs.size() == 2 ? cur->inputs[1] : nullptr;
        auto a_requires_grad = a->flags & NodeFlags::REQUIRES_GRAD;
        auto b_requires_grad = b ? b->flags & NodeFlags::REQUIRES_GRAD : 0;

        if(!a_requires_grad){
            if(!b)
                continue;
            if(!b_requires_grad)
                continue;
        }
        bool res = true;
        switch(cur->operation){
            case Op::NONE:
            case Op::CREATE: break;

            case Op::UNARY_START: break;

            case Op::RELU: {
                // g(a) = max(0, a)
                // df/da += df/dg . dg / da = df/dg if a > 0 else 0
                res &= a_requires_grad ? reluMatAddGradient(&a->grad, &a->value, &cur->grad) : true;
            } break;
            case Op::SOFTMAX: {
                // g(a) = softmax(a), a -> Rn
                // dg/da = J -> Rnxn where Jij = softmax(ai) * ((i == j) - softmax(aj))
                // df/da += dg/da @ df/dg 
                res &= a_requires_grad ? softmaxMatAddGradient(&a->grad, &cur->value, &cur->grad) : true;
            } break;

            case Op::BINARY_START: break;

            case Op::ADD: {
                if(b == nullptr) {res = false; break;}
                // g(a,b) = a + b
                // df/da += df/dg . dg/da = df/dg (cur->grad)
                // df/db += df/dg . dg/db = df/dg
                res &= a_requires_grad ? addMat(&a->grad, &a->grad, &cur->grad) : true;
                res &= b_requires_grad ? addMat(&b->grad, &b->grad, &cur->grad) : true;
            } break;
            case Op::SUB: {
                if(b == nullptr) {res = false; break;}
                // g(a,b) = a - b
                // df/da += df/dg . dg/da = df/dg (cur->grad)
                // df/db += df/dg . dg/db = -df/dg
                res &= a_requires_grad ? addMat(&a->grad, &a->grad, &cur->grad) : true;
                res &= b_requires_grad ? subMat(&b->grad, &b->grad, &cur->grad) : true;
            } break;
            case Op::MATMUL: {
                if(b == nullptr) {res = false; break;}
                // g(a, b) = a @ b
                // df/da += df/dg . dg/da = df/dg @ bT
                // df/db += df/dg . dg/db = aT @ df/dg
                res &= a_requires_grad ? mulMat(&a->grad , &cur->grad , &b->value, 0, 0, 1) : true;    
                res &= b_requires_grad ? mulMat(&b->grad, &a->value, &cur->grad, 0, 1, 0) : true;
            } break;
            case Op::CROSS_ENTROPY: {
                if(b == nullptr) {res = false; break;}
                // g(a, b) = -b . ln(a)
                // df/da += df/dg . dg/da = df/dg . - b / a
                // df/db += df/dg . dg/db = df/dg . -ln(a)
                res &= crossEntropyMatAddGradient(&a->value,&b->value, a_requires_grad ? &a->grad : nullptr,
                     b_requires_grad ? &b->grad : nullptr, &cur->grad);
            } break;

        }

    }

}

void modelContextCompile(ModelContext* ctx) {
    if(ctx->output) {
        ctx->forward_pass = graphCreate(ctx,ctx->output);
    }
    if(ctx->loss) {
        ctx->loss_fn = graphCreate(ctx, ctx->loss);
    }
}

void modelContextFeedForward(ModelContext* ctx) {
    graphCompute(&ctx->forward_pass);
}




