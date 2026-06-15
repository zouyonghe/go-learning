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

On macOS, the experimental local Gorgonia checkout can train the same MLP with MPSGraph-backed forward and loss. `ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26` is currently required by Gorgonia's `go4.org/unsafe/assume-no-moving-gc` dependency when running with Go 1.26.

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

For the standard Gorgonia `Grad + Solver` path, use `train-mps-solver`:

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

The saved checkpoint uses the same `.gob` format as CPU training, so the regular `infer` command can load it.

See [docs/mpsgraph.md](docs/mpsgraph.md) for backend architecture, supported ops, smoke commands, verification steps, and upstream PR readiness notes.

## Infer

Use a PNG or JPEG digit image. The image is converted to grayscale and resized to 28x28.

```bash
go run ./cmd/mnist infer \
  -model ./mnist.gob \
  -image ./digit.png \
  -topk 3
```

## Notes

- This is a learning-oriented implementation, not a production training framework.
- Full MNIST CPU training is feasible, but accuracy depends on epochs and learning rate.
- Gorgonia can express computation graphs and automatic differentiation; the CPU example keeps backpropagation explicit, while the MPS solver path exercises `Grad` and Gorgonia solvers.
