//go:build mps && darwin
// +build mps,darwin

package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strings"

	"go-learning/internal/mnist"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func trainMPSSolver(args []string) {
	fs := flag.NewFlagSet("train-mps-solver", flag.ExitOnError)
	dataDir := fs.String("data", "./data", "directory containing MNIST IDX files")
	epochs := fs.Int("epochs", 1, "number of training epochs")
	batchSize := fs.Int("batch", 32, "mini-batch size")
	lr := fs.Float64("lr", 0.001, "learning rate")
	optimizer := fs.String("optimizer", "adam", "optimizer: sgd or adam")
	modelPath := fs.String("model", "./mnist.gob", "checkpoint path")
	hidden := fs.Int("hidden", 128, "hidden layer size")
	limit := fs.Int("limit", 1000, "optional max training samples, 0 means all")
	must(fs.Parse(args))

	set, err := mnist.LoadIDXDataset(*dataDir, true)
	must(err)
	if *limit > 0 && *limit < len(set.Images) {
		set.Images = set.Images[:*limit]
		set.Labels = set.Labels[:*limit]
	}

	model := newMPSMLP(28*28, *hidden, 10)
	model.trainWithSolver(set, *epochs, *batchSize, float32(*lr), *optimizer)
	must(os.MkdirAll(filepath.Dir(*modelPath), 0o755))
	must(model.toMLP().Save(*modelPath))
	fmt.Printf("saved model to %s\n", *modelPath)
}

func randPerm(n int) []int { return rand.Perm(n) }

func shuffleIndices(indices []int) {
	rand.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
}

func (m *mpsMLP) trainWithSolver(data mnist.Dataset, epochs, batchSize int, lr float32, optimizerName string) []mnist.EpochStat {
	if epochs <= 0 {
		log.Fatal("epochs must be positive")
	}
	if batchSize <= 0 {
		log.Fatal("batch size must be positive")
	}
	if lr <= 0 {
		log.Fatal("learning rate must be positive")
	}
	if len(data.Images) == 0 {
		log.Fatal("empty dataset")
	}

	stats := make([]mnist.EpochStat, 0, epochs)
	indices := randPerm(len(data.Images))
	solver := newGorgoniaSolver(optimizerName, lr)
	for epoch := 1; epoch <= epochs; epoch++ {
		shuffleIndices(indices)
		var totalLoss float64
		var correct int
		for start := 0; start < len(indices); start += batchSize {
			end := start + batchSize
			if end > len(indices) {
				end = len(indices)
			}
			loss, batchCorrect := m.trainBatchWithSolver(data, indices[start:end], solver)
			totalLoss += float64(loss) * float64(end-start)
			correct += batchCorrect
		}
		stat := mnist.EpochStat{Epoch: epoch, Loss: totalLoss / float64(len(data.Images)), Accuracy: float64(correct) / float64(len(data.Images))}
		stats = append(stats, stat)
		fmt.Printf("epoch=%d loss=%.4f accuracy=%.2f%%\n", stat.Epoch, stat.Loss, stat.Accuracy*100)
	}
	return stats
}

func (m *mpsMLP) trainBatchWithSolver(data mnist.Dataset, batch []int, solver G.Solver) (float32, int) {
	xBatch := make([]float32, len(batch)*m.inputSize)
	labels := make([]int32, len(batch))
	for row, idx := range batch {
		for col, value := range data.Images[idx] {
			xBatch[row*m.inputSize+col] = float32(value)
		}
		labels[row] = int32(data.Labels[idx])
	}

	g := G.NewGraph()
	x := G.NewMatrix(g, tensor.Float32, G.WithShape(len(batch), m.inputSize), G.WithName("x"))
	w1 := G.NewMatrix(g, tensor.Float32, G.WithShape(m.inputSize, m.hiddenSize), G.WithName("w1"))
	b1 := G.NewVector(g, tensor.Float32, G.WithShape(m.hiddenSize), G.WithName("b1"))
	w2 := G.NewMatrix(g, tensor.Float32, G.WithShape(m.hiddenSize, m.classes), G.WithName("w2"))
	b2 := G.NewVector(g, tensor.Float32, G.WithShape(m.classes), G.WithName("b2"))
	labelNode := G.NewVector(g, tensor.Int32, G.WithShape(len(batch)), G.WithName("labels"))

	l1, err := G.Mul(x, w1)
	must(err)
	l1Bias, err := G.BroadcastAdd(l1, b1, nil, []byte{0})
	must(err)
	hidden, err := G.Rectify(l1Bias)
	must(err)
	logits, err := G.Mul(hidden, w2)
	must(err)
	logitsBias, err := G.BroadcastAdd(logits, b2, nil, []byte{0})
	must(err)
	loss, err := G.MPSCrossEntropy(logitsBias, labelNode)
	must(err)
	if _, err := G.Grad(loss, w1, b1, w2, b2); err != nil {
		must(err)
	}

	machine := G.NewTapeMachine(g, G.BindDualValues(w1, b1, w2, b2))
	defer machine.Close()
	must(G.Let(x, tensor.New(tensor.WithShape(len(batch), m.inputSize), tensor.WithBacking(append([]float32(nil), xBatch...)))))
	must(G.Let(w1, tensor.New(tensor.WithShape(m.inputSize, m.hiddenSize), tensor.WithBacking(append([]float32(nil), m.w1...)))))
	must(G.Let(b1, tensor.New(tensor.WithShape(m.hiddenSize), tensor.WithBacking(append([]float32(nil), m.b1...)))))
	must(G.Let(w2, tensor.New(tensor.WithShape(m.hiddenSize, m.classes), tensor.WithBacking(append([]float32(nil), m.w2...)))))
	must(G.Let(b2, tensor.New(tensor.WithShape(m.classes), tensor.WithBacking(append([]float32(nil), m.b2...)))))
	must(G.Let(labelNode, tensor.New(tensor.WithShape(len(batch)), tensor.WithBacking(append([]int32(nil), labels...)))))
	must(machine.RunAll())

	must(solver.Step([]G.ValueGrad{w1, b1, w2, b2}))
	copy(m.w1, w1.Value().(*tensor.Dense).Data().([]float32))
	copy(m.b1, b1.Value().(*tensor.Dense).Data().([]float32))
	copy(m.w2, w2.Value().(*tensor.Dense).Data().([]float32))
	copy(m.b2, b2.Value().(*tensor.Dense).Data().([]float32))
	return loss.Value().(*G.F32).Data().(float32), countCorrect(logitsBias.Value().(*tensor.Dense).Data().([]float32), labels, m.classes)
}

func newGorgoniaSolver(name string, lr float32) G.Solver {
	switch strings.ToLower(name) {
	case "sgd":
		return G.NewVanillaSolver(G.WithLearnRate(float64(lr)))
	case "adam":
		return G.NewAdamSolver(G.WithLearnRate(float64(lr)))
	default:
		log.Fatalf("unsupported optimizer %q; use sgd or adam", name)
		return nil
	}
}
