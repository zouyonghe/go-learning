# Go MNIST Digit Recognizer

This project trains and runs inference for handwritten digits in Go. It uses Gorgonia's tensor package as part of a small neural-network example and keeps the training loop explicit so the math is easy to inspect.

The model is a multilayer perceptron:

```text
784 input pixels -> 128 ReLU hidden units -> 10 digit logits -> softmax
```

## Install Dependencies

```bash
go mod tidy
```

## Download MNIST

Create a `data` directory and download the IDX files:

```bash
mkdir -p data
curl -L -o data/train-images-idx3-ubyte.gz https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -L -o data/train-labels-idx1-ubyte.gz https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -L -o data/t10k-images-idx3-ubyte.gz https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -L -o data/t10k-labels-idx1-ubyte.gz https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
```

The loader accepts either `.gz` files or uncompressed IDX files.

## Train

```bash
go run ./cmd/mnist train \
  -data ./data \
  -epochs 5 \
  -batch 64 \
  -lr 0.01 \
  -model ./mnist.gob
```

For a quick smoke test, train on a small subset:

```bash
go run ./cmd/mnist train -data ./data -epochs 1 -batch 32 -limit 1000 -model ./mnist.gob
```

## Train With Apple MPSGraph

On macOS, the experimental local Gorgonia checkout can train the same MLP with MPSGraph-backed forward, cross-entropy loss, logits gradient, and hand-written two-layer updates. `ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26` is currently required by Gorgonia's `go4.org/unsafe/assume-no-moving-gc` dependency when running with Go 1.26.

```bash
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go run -tags mps ./cmd/mnist train-mps \
  -data ./data \
  -epochs 1 \
  -batch 32 \
  -limit 1000 \
  -lr 0.01 \
  -optimizer sgd \
  -model ./mnist.gob
```

Adam is also available and usually converges faster on small subsets:

```bash
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go run -tags mps ./cmd/mnist train-mps \
  -data ./data \
  -epochs 10 \
  -batch 32 \
  -limit 1000 \
  -lr 0.001 \
  -optimizer adam \
  -model ./mnist-mps.gob
```

The saved checkpoint uses the same `.gob` format as CPU training, so the regular `infer` command can load it.

## Infer

Use a PNG or JPEG digit image. The image is converted to grayscale and resized to 28x28.

```bash
go run ./cmd/mnist infer \
  -model ./mnist.gob \
  -image ./digit.png \
  -topk 3
```

## Apple MPS Smoke Test

This repository is wired to the local experimental Gorgonia checkout with:

```go
replace gorgonia.org/gorgonia => ../gorgonia
```

Run a small `XW + b` forward pass through the experimental MPSGraph backend:

```bash
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go run -tags mps ./cmd/mps-smoke
```

Expected output:

```text
[32 48 59 84]
```

Run a small two-layer MLP forward pass through MPSGraph-backed Gorgonia ops:

```bash
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go run -tags mps ./cmd/mps-mlp-forward
```

Expected output:

```text
[2.25 21.25 2.25 48.25]
```

Run a small row-wise softmax pass through MPSGraph-backed Gorgonia ops:

```bash
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go run -tags mps ./cmd/mps-softmax
```

Expected output:

```text
[0.0900306 0.244728 0.665241 0.333333 0.333333 0.333333]
```

Run a small row-wise log-softmax pass through MPSGraph-backed Gorgonia ops:

```bash
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go run -tags mps ./cmd/mps-log-softmax
```

Expected output:

```text
[-2.407606 -1.407606 -0.407606 -1.098612 -1.098612 -1.098612]
```

Run the two-layer MLP through MPSGraph-backed logits, log-softmax, and NLL loss:

```bash
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go run -tags mps ./cmd/mps-mlp-loss
```

Expected output:

```text
23.000000
```

Run one hand-written SGD step for the final MLP layer using MPSGraph-backed loss and logits gradient:

```bash
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go run -tags mps ./cmd/mps-one-step-sgd
```

Expected output:

```text
before=23.000000 after=20.424999
```

Run one hand-written SGD step for both MLP layers using the same MPSGraph-backed forward/loss path:

```bash
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go run -tags mps ./cmd/mps-two-layer-sgd
```

Expected output:

```text
before=23.000000 after=19.541649
```

## Notes

- This is a learning-oriented implementation, not a production training framework.
- Full MNIST CPU training is feasible, but accuracy depends on epochs and learning rate.
- Gorgonia can express computation graphs and automatic differentiation; this sample keeps backpropagation explicit to make the network easier to follow and more stable across library versions.
