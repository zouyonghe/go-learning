# Apple MPSGraph Backend Notes

This project uses a local experimental Gorgonia checkout with Apple MPSGraph support:

```go
replace gorgonia.org/gorgonia => ../gorgonia
```

The implementation targets macOS Apple GPU execution through Metal/MPSGraph. It is useful for learning and backend validation, but it is not yet a production-quality Gorgonia backend.

## Runtime Requirements

- macOS with a Metal-capable Apple GPU.
- Go 1.26 currently requires:

```bash
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26
```

This is required by Gorgonia's `go4.org/unsafe/assume-no-moving-gc` dependency.

## Supported Gorgonia Paths

The forked Gorgonia branch currently supports these MPS-backed paths:

- `Mul` for non-transposed 2D `float32` matrix multiplication.
- Tensor elementwise `Add`, `Sub`, and `HadamardProd` for same-shaped `float32` tensors.
- `BroadcastAdd(matrix, bias, nil, []byte{0})` for row bias add.
- `Rectify` for `float32` tensors via MPSGraph ReLU.
- `SoftMax` and `LogSoftMax` for 2D `float32` row-wise classification logits.
- `MPSCrossEntropy(logits, labels)` for `float32` logits and `int32` labels.
- `Grad(MPSCrossEntropy(...), logits)` and `Grad(loss, w1, b1, w2, b2)` for the two-layer MLP path.

Unsupported ops fall back to CPU when they are not tagged as MPS-supported.

## MNIST Training Modes

### Hand-Written MPS Training

`train-mps` uses MPSGraph-backed forward/loss and a hand-written two-layer backward pass:

```bash
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go run -tags mps ./cmd/mnist train-mps \
  -data ./data \
  -epochs 10 \
  -batch 32 \
  -limit 1000 \
  -hidden 128 \
  -lr 0.001 \
  -optimizer adam \
  -model ./mnist-mps.gob
```

### Standard Grad/Solver MPS Training

`train-mps-solver` builds the same MPSGraph-backed forward graph, calls `Grad(loss, w1, b1, w2, b2)`, and updates parameters with a Gorgonia solver:

```bash
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go run -tags mps ./cmd/mnist train-mps-solver \
  -data ./data \
  -epochs 10 \
  -batch 32 \
  -limit 1000 \
  -hidden 128 \
  -lr 0.001 \
  -optimizer adam \
  -model ./mnist-mps-solver.gob
```

`train-mps-solver` keeps the solver instance alive across batches, so Adam momentum and variance state are preserved during training.

Both modes save the same `.gob` checkpoint format as CPU training, so `go run ./cmd/mnist infer ...` can load the result.

## Smoke Commands

Use these commands to validate individual backend slices:

```bash
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go run -tags mps ./cmd/mps-smoke
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go run -tags mps ./cmd/mps-mlp-forward
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go run -tags mps ./cmd/mps-softmax
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go run -tags mps ./cmd/mps-log-softmax
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go run -tags mps ./cmd/mps-mlp-loss
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go run -tags mps ./cmd/mps-one-step-sgd
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go run -tags mps ./cmd/mps-two-layer-sgd
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go run -tags mps ./cmd/mps-solver-step
```

Expected representative outputs:

```text
./cmd/mps-smoke          -> [32 48 59 84]
./cmd/mps-mlp-forward    -> [2.25 21.25 2.25 48.25]
./cmd/mps-softmax        -> [0.0900306 0.244728 0.665241 0.333333 0.333333 0.333333]
./cmd/mps-log-softmax    -> [-2.407606 -1.407606 -0.407606 -1.098612 -1.098612 -1.098612]
./cmd/mps-mlp-loss       -> 23.000000
./cmd/mps-one-step-sgd   -> before=23.000000 after=20.424999
./cmd/mps-two-layer-sgd  -> before=23.000000 after=19.541649
./cmd/mps-solver-step    -> before=23.000000 after=22.433197
```

## Verification

Run project tests:

```bash
rtk go test ./...
```

Run the Gorgonia MPS test suite from the local fork:

```bash
cd ../gorgonia
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 rtk go test -tags mps .
```

To visually confirm GPU use, open Activity Monitor and select `Window -> GPU History` while running an MPS training command.

## Current Limitations

- Tensor data does not stay resident on GPU across ops; outputs return as CPU-accessible tensors.
- MPSGraph execution currently creates small per-op graphs instead of a fused compiled graph.
- The backend supports only a narrow `float32` MLP/classification path.
- Transformer-oriented ops are not complete yet: batched matmul, attention masking, layer norm, GELU, embedding/gather, and shape-heavy graph transforms need more work.
- `MPSCrossEntropy` is still an experimental API. A future upstream-ready version should align better with Gorgonia's general loss API naming and autodiff conventions.

## Upstream PR Readiness

The fork is not ready for a single upstream PR as-is. The current diff is best treated as an experimental backend branch and split into smaller reviewable PRs:

1. Device/build-tag/VM extension points.
2. MPS bridge runtime and low-level op tests.
3. MatMul/elementwise/ReLU integration through existing Gorgonia APIs.
4. SoftMax/LogSoftMax and loss/autodiff integration.
5. Documentation, benchmarks, and backend support matrix.

The most important next backend improvements are GPU-resident buffers, broader op coverage, and a clearer public loss API.
