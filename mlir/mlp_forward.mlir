// A small, real 2-layer MLP forward pass, hand-authored directly in
// MLIR's linalg/arith/tensor dialects — not generated from a
// higher-level framework. This is the same IR level (linalg-on-tensor)
// that a real ML-compiler frontend (e.g. torch-mlir) would lower a
// PyTorch model down to before target-specific optimization; writing
// it by hand here is an honest, deliberately scoped-down entry point
// into that space, not a claim of having built a frontend.
//
// Network: input (1x4) -> Linear(4,6) -> ReLU -> Linear(6,3) -> output (1x3)
//   layer1_out = relu(x @ W1 + b1)
//   output     = layer1_out @ W2 + b2
//
// linalg.matmul and linalg.generic (for the bias-add + ReLU, which
// isn't a named linalg op) are genuine MLIR ops — this file is valid
// input to mlir-opt and is actually lowered and executed, not just
// inspected. See run_pipeline.sh for the real lowering commands and
// tests/test_mlp_reference.py for the NumPy reference this is checked
// against.

func.func @mlp_forward(
    %x: tensor<1x4xf32>,
    %w1: tensor<4x6xf32>,
    %b1: tensor<1x6xf32>,
    %w2: tensor<6x3xf32>,
    %b2: tensor<1x3xf32>
  ) -> tensor<1x3xf32> {

  // --- Layer 1: x @ W1 ---
  %zero6 = arith.constant 0.0 : f32
  %init1 = tensor.empty() : tensor<1x6xf32>
  %fill1 = linalg.fill ins(%zero6 : f32) outs(%init1 : tensor<1x6xf32>) -> tensor<1x6xf32>
  %mm1 = linalg.matmul ins(%x, %w1 : tensor<1x4xf32>, tensor<4x6xf32>)
                        outs(%fill1 : tensor<1x6xf32>) -> tensor<1x6xf32>

  // --- Layer 1: + b1, then ReLU (fused elementwise via linalg.generic) ---
  %relu1 = linalg.generic {
      indexing_maps = [
        affine_map<(i, j) -> (i, j)>,
        affine_map<(i, j) -> (i, j)>,
        affine_map<(i, j) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%mm1, %b1 : tensor<1x6xf32>, tensor<1x6xf32>)
      outs(%init1 : tensor<1x6xf32>) {
    ^bb0(%mm_val: f32, %bias_val: f32, %out: f32):
      %sum = arith.addf %mm_val, %bias_val : f32
      %relu = arith.maximumf %sum, %zero6 : f32
      linalg.yield %relu : f32
  } -> tensor<1x6xf32>

  // --- Layer 2: relu1 @ W2 ---
  %zero3 = arith.constant 0.0 : f32
  %init2 = tensor.empty() : tensor<1x3xf32>
  %fill2 = linalg.fill ins(%zero3 : f32) outs(%init2 : tensor<1x3xf32>) -> tensor<1x3xf32>
  %mm2 = linalg.matmul ins(%relu1, %w2 : tensor<1x6xf32>, tensor<6x3xf32>)
                        outs(%fill2 : tensor<1x3xf32>) -> tensor<1x3xf32>

  // --- Layer 2: + b2 (no activation on output layer) ---
  %out = linalg.generic {
      indexing_maps = [
        affine_map<(i, j) -> (i, j)>,
        affine_map<(i, j) -> (i, j)>,
        affine_map<(i, j) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%mm2, %b2 : tensor<1x3xf32>, tensor<1x3xf32>)
      outs(%init2 : tensor<1x3xf32>) {
    ^bb0(%mm_val: f32, %bias_val: f32, %o: f32):
      %sum = arith.addf %mm_val, %bias_val : f32
      linalg.yield %sum : f32
  } -> tensor<1x3xf32>

  return %out : tensor<1x3xf32>
}
