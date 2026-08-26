// Executable driver around mlp_forward (see mlp_forward.mlir for the
// network itself and its documentation). This file builds concrete
// constant tensors for the input and weights, calls the network, and
// prints each output element via the real MLIR C-runner-utils
// printF32/printNewline functions — so mlir-cpu-runner produces actual
// numeric output that gets checked against a NumPy reference
// (tests/test_mlp_reference.py), not just "the pipeline didn't crash."
//
// Weight/bias values here match tests/test_mlp_reference.py exactly —
// see that file for the single source of truth and the exact expected
// output values.

func.func private @printF32(f32)
func.func private @printNewline()

func.func @mlp_forward(
    %x: tensor<1x4xf32>,
    %w1: tensor<4x6xf32>,
    %b1: tensor<1x6xf32>,
    %w2: tensor<6x3xf32>,
    %b2: tensor<1x3xf32>
  ) -> tensor<1x3xf32> {
  %zero6 = arith.constant 0.0 : f32
  %init1 = tensor.empty() : tensor<1x6xf32>
  %fill1 = linalg.fill ins(%zero6 : f32) outs(%init1 : tensor<1x6xf32>) -> tensor<1x6xf32>
  %mm1 = linalg.matmul ins(%x, %w1 : tensor<1x4xf32>, tensor<4x6xf32>)
                        outs(%fill1 : tensor<1x6xf32>) -> tensor<1x6xf32>

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

  %zero3 = arith.constant 0.0 : f32
  %init2 = tensor.empty() : tensor<1x3xf32>
  %fill2 = linalg.fill ins(%zero3 : f32) outs(%init2 : tensor<1x3xf32>) -> tensor<1x3xf32>
  %mm2 = linalg.matmul ins(%relu1, %w2 : tensor<1x6xf32>, tensor<6x3xf32>)
                        outs(%fill2 : tensor<1x3xf32>) -> tensor<1x3xf32>

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

func.func @main() {
  // Input: x = [1.0, -2.0, 0.5, 3.0]
  %x = arith.constant dense<[[1.0, -2.0, 0.5, 3.0]]> : tensor<1x4xf32>

  // W1: 4x6, hand-picked fixed values (matches the NumPy reference).
  %w1 = arith.constant dense<[
    [0.1, 0.2, -0.1, 0.05, 0.3, -0.2],
    [0.4, -0.3, 0.2, 0.1, -0.1, 0.05],
    [-0.2, 0.1, 0.3, -0.4, 0.2, 0.1],
    [0.05, 0.15, -0.25, 0.2, -0.1, 0.3]
  ]> : tensor<4x6xf32>
  %b1 = arith.constant dense<[[0.01, -0.02, 0.03, 0.0, -0.01, 0.02]]> : tensor<1x6xf32>

  // W2: 6x3
  %w2 = arith.constant dense<[
    [0.2, -0.1, 0.3],
    [-0.3, 0.2, 0.1],
    [0.1, 0.4, -0.2],
    [0.05, -0.05, 0.15],
    [-0.2, 0.3, 0.1],
    [0.15, -0.1, 0.2]
  ]> : tensor<6x3xf32>
  %b2 = arith.constant dense<[[0.1, -0.1, 0.05]]> : tensor<1x3xf32>

  %result = func.call @mlp_forward(%x, %w1, %b1, %w2, %b2)
    : (tensor<1x4xf32>, tensor<4x6xf32>, tensor<1x6xf32>, tensor<6x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %e0 = tensor.extract %result[%c0, %c0] : tensor<1x3xf32>
  %e1 = tensor.extract %result[%c0, %c1] : tensor<1x3xf32>
  %e2 = tensor.extract %result[%c0, %c2] : tensor<1x3xf32>

  func.call @printF32(%e0) : (f32) -> ()
  func.call @printNewline() : () -> ()
  func.call @printF32(%e1) : (f32) -> ()
  func.call @printNewline() : () -> ()
  func.call @printF32(%e2) : (f32) -> ()
  func.call @printNewline() : () -> ()

  return
}
