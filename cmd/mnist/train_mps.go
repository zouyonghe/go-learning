//go:build mps && darwin
// +build mps,darwin

package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	"go-learning/internal/mnist"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type mpsMLP struct {
	inputSize  int
	hiddenSize int
	classes    int
	w1         []float32
	b1         []float32
	w2         []float32
	b2         []float32
}

type mpsBatchResult struct {
	loss    float32
	preRelu []float32
	hidden  []float32
	logits  []float32
	dLogits []float32
}

func trainMPS(args []string) {
	fs := flag.NewFlagSet("train-mps", flag.ExitOnError)
	dataDir := fs.String("data", "./data", "directory containing MNIST IDX files")
	epochs := fs.Int("epochs", 1, "number of training epochs")
	batchSize := fs.Int("batch", 32, "mini-batch size")
	lr := fs.Float64("lr", 0.01, "learning rate")
	optimizer := fs.String("optimizer", "sgd", "optimizer: sgd or adam")
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
	model.train(set, *epochs, *batchSize, float32(*lr), *optimizer)

	must(os.MkdirAll(filepath.Dir(*modelPath), 0o755))
	must(model.toMLP().Save(*modelPath))
	fmt.Printf("saved model to %s\n", *modelPath)
}

func newMPSMLP(inputSize, hiddenSize, classes int) *mpsMLP {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	model := &mpsMLP{
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		classes:    classes,
		w1:         make([]float32, inputSize*hiddenSize),
		b1:         make([]float32, hiddenSize),
		w2:         make([]float32, hiddenSize*classes),
		b2:         make([]float32, classes),
	}
	for i := range model.w1 {
		model.w1[i] = float32(rng.NormFloat64() * math.Sqrt(2.0/float64(inputSize)))
	}
	for i := range model.w2 {
		model.w2[i] = float32(rng.NormFloat64() * math.Sqrt(2.0/float64(hiddenSize)))
	}
	return model
}

func (m *mpsMLP) train(data mnist.Dataset, epochs, batchSize int, lr float32, optimizerName string) []mnist.EpochStat {
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
	indices := rand.Perm(len(data.Images))
	optimizer := newMPSOptimizer(optimizerName, lr, m)
	for epoch := 1; epoch <= epochs; epoch++ {
		rand.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })

		var totalLoss float64
		var correct int
		for start := 0; start < len(indices); start += batchSize {
			end := start + batchSize
			if end > len(indices) {
				end = len(indices)
			}
			loss, batchCorrect := m.trainBatch(data, indices[start:end], optimizer)
			totalLoss += float64(loss) * float64(end-start)
			correct += batchCorrect
		}

		stat := mnist.EpochStat{
			Epoch:    epoch,
			Loss:     totalLoss / float64(len(data.Images)),
			Accuracy: float64(correct) / float64(len(data.Images)),
		}
		stats = append(stats, stat)
		fmt.Printf("epoch=%d loss=%.4f accuracy=%.2f%%\n", stat.Epoch, stat.Loss, stat.Accuracy*100)
	}
	return stats
}

func (m *mpsMLP) trainBatch(data mnist.Dataset, batch []int, optimizer *mpsOptimizer) (float32, int) {
	xBatch := make([]float32, len(batch)*m.inputSize)
	labels := make([]int32, len(batch))
	for row, idx := range batch {
		for col, v := range data.Images[idx] {
			xBatch[row*m.inputSize+col] = float32(v)
		}
		labels[row] = int32(data.Labels[idx])
	}

	result := m.forwardBatch(xBatch, labels, len(batch))
	dW1 := make([]float32, len(m.w1))
	dB1 := make([]float32, len(m.b1))
	dW2 := make([]float32, len(m.w2))
	dB2 := make([]float32, len(m.b2))

	dHidden := make([]float32, len(batch)*m.hiddenSize)
	for row := 0; row < len(batch); row++ {
		for hiddenCol := 0; hiddenCol < m.hiddenSize; hiddenCol++ {
			var grad float32
			for classCol := 0; classCol < m.classes; classCol++ {
				grad += result.dLogits[row*m.classes+classCol] * m.w2[hiddenCol*m.classes+classCol]
			}
			dHidden[row*m.hiddenSize+hiddenCol] = grad
		}
	}

	dPreRelu := make([]float32, len(dHidden))
	for i := range dHidden {
		if result.preRelu[i] > 0 {
			dPreRelu[i] = dHidden[i]
		}
	}

	for hiddenCol := 0; hiddenCol < m.hiddenSize; hiddenCol++ {
		for classCol := 0; classCol < m.classes; classCol++ {
			var grad float32
			for row := 0; row < len(batch); row++ {
				grad += result.hidden[row*m.hiddenSize+hiddenCol] * result.dLogits[row*m.classes+classCol]
			}
			dW2[hiddenCol*m.classes+classCol] = grad
		}
	}
	for classCol := 0; classCol < m.classes; classCol++ {
		var grad float32
		for row := 0; row < len(batch); row++ {
			grad += result.dLogits[row*m.classes+classCol]
		}
		dB2[classCol] = grad
	}

	for inputCol := 0; inputCol < m.inputSize; inputCol++ {
		for hiddenCol := 0; hiddenCol < m.hiddenSize; hiddenCol++ {
			var grad float32
			for row := 0; row < len(batch); row++ {
				grad += xBatch[row*m.inputSize+inputCol] * dPreRelu[row*m.hiddenSize+hiddenCol]
			}
			dW1[inputCol*m.hiddenSize+hiddenCol] = grad
		}
	}
	for hiddenCol := 0; hiddenCol < m.hiddenSize; hiddenCol++ {
		var grad float32
		for row := 0; row < len(batch); row++ {
			grad += dPreRelu[row*m.hiddenSize+hiddenCol]
		}
		dB1[hiddenCol] = grad
	}
	optimizer.step(m, dW1, dB1, dW2, dB2)

	return result.loss, countCorrect(result.logits, labels, m.classes)
}

type mpsOptimizer struct {
	name string
	lr   float32
	b1   float32
	b2   float32
	eps  float32
	stepCount int
	mw1, vw1 []float32
	mb1, vb1 []float32
	mw2, vw2 []float32
	mb2, vb2 []float32
}

func newMPSOptimizer(name string, lr float32, model *mpsMLP) *mpsOptimizer {
	opt := &mpsOptimizer{name: strings.ToLower(name), lr: lr, b1: 0.9, b2: 0.999, eps: 1e-8}
	switch opt.name {
	case "sgd":
		return opt
	case "adam":
		opt.mw1, opt.vw1 = make([]float32, len(model.w1)), make([]float32, len(model.w1))
		opt.mb1, opt.vb1 = make([]float32, len(model.b1)), make([]float32, len(model.b1))
		opt.mw2, opt.vw2 = make([]float32, len(model.w2)), make([]float32, len(model.w2))
		opt.mb2, opt.vb2 = make([]float32, len(model.b2)), make([]float32, len(model.b2))
		return opt
	default:
		log.Fatalf("unsupported optimizer %q; use sgd or adam", name)
		return nil
	}
}

func (o *mpsOptimizer) step(model *mpsMLP, dW1, dB1, dW2, dB2 []float32) {
	o.stepCount++
	switch o.name {
	case "sgd":
		applySGD(model.w1, dW1, o.lr)
		applySGD(model.b1, dB1, o.lr)
		applySGD(model.w2, dW2, o.lr)
		applySGD(model.b2, dB2, o.lr)
	case "adam":
		applyAdam(model.w1, dW1, o.mw1, o.vw1, o)
		applyAdam(model.b1, dB1, o.mb1, o.vb1, o)
		applyAdam(model.w2, dW2, o.mw2, o.vw2, o)
		applyAdam(model.b2, dB2, o.mb2, o.vb2, o)
	}
}

func applySGD(params, grads []float32, lr float32) {
	for i := range params {
		params[i] -= lr * grads[i]
	}
}

func applyAdam(params, grads, moments, velocities []float32, opt *mpsOptimizer) {
	b1Correction := float32(1 - math.Pow(float64(opt.b1), float64(opt.stepCount)))
	b2Correction := float32(1 - math.Pow(float64(opt.b2), float64(opt.stepCount)))
	for i := range params {
		moments[i] = opt.b1*moments[i] + (1-opt.b1)*grads[i]
		velocities[i] = opt.b2*velocities[i] + (1-opt.b2)*grads[i]*grads[i]
		mHat := moments[i] / b1Correction
		vHat := velocities[i] / b2Correction
		params[i] -= opt.lr * mHat / (float32(math.Sqrt(float64(vHat))) + opt.eps)
	}
}

func (m *mpsMLP) forwardBatch(xBatch []float32, labels []int32, batchSize int) mpsBatchResult {
	g := G.NewGraph()
	x := G.NewMatrix(g, tensor.Float32, G.WithShape(batchSize, m.inputSize), G.WithName("x"))
	w1 := G.NewMatrix(g, tensor.Float32, G.WithShape(m.inputSize, m.hiddenSize), G.WithName("w1"))
	b1 := G.NewVector(g, tensor.Float32, G.WithShape(m.hiddenSize), G.WithName("b1"))
	w2 := G.NewMatrix(g, tensor.Float32, G.WithShape(m.hiddenSize, m.classes), G.WithName("w2"))
	b2 := G.NewVector(g, tensor.Float32, G.WithShape(m.classes), G.WithName("b2"))
	labelNode := G.NewVector(g, tensor.Int32, G.WithShape(batchSize), G.WithName("labels"))

	l1, err := G.Mul(x, w1)
	must(err)
	preRelu, err := G.BroadcastAdd(l1, b1, nil, []byte{0})
	must(err)
	hidden, err := G.Rectify(preRelu)
	must(err)
	logits, err := G.Mul(hidden, w2)
	must(err)
	logitsBias, err := G.BroadcastAdd(logits, b2, nil, []byte{0})
	must(err)
	loss, err := G.MPSCrossEntropy(logitsBias, labelNode)
	must(err)

	machine := G.NewTapeMachine(g)
	defer machine.Close()
	must(G.Let(x, tensor.New(tensor.WithShape(batchSize, m.inputSize), tensor.WithBacking(append([]float32(nil), xBatch...)))))
	must(G.Let(w1, tensor.New(tensor.WithShape(m.inputSize, m.hiddenSize), tensor.WithBacking(append([]float32(nil), m.w1...)))))
	must(G.Let(b1, tensor.New(tensor.WithShape(m.hiddenSize), tensor.WithBacking(append([]float32(nil), m.b1...)))))
	must(G.Let(w2, tensor.New(tensor.WithShape(m.hiddenSize, m.classes), tensor.WithBacking(append([]float32(nil), m.w2...)))))
	must(G.Let(b2, tensor.New(tensor.WithShape(m.classes), tensor.WithBacking(append([]float32(nil), m.b2...)))))
	must(G.Let(labelNode, tensor.New(tensor.WithShape(batchSize), tensor.WithBacking(append([]int32(nil), labels...)))))
	must(machine.RunAll())

	logitsData := append([]float32(nil), logitsBias.Value().(*tensor.Dense).Data().([]float32)...)
	return mpsBatchResult{
		loss:    loss.Value().(*G.F32).Data().(float32),
		preRelu: append([]float32(nil), preRelu.Value().(*tensor.Dense).Data().([]float32)...),
		hidden:  append([]float32(nil), hidden.Value().(*tensor.Dense).Data().([]float32)...),
		logits:  logitsData,
		dLogits: crossEntropyLogitsGrad(logitsData, labels, batchSize, m.classes),
	}
}

func crossEntropyLogitsGrad(logitsData []float32, labelsData []int32, rows, classes int) []float32 {
	g := G.NewGraph()
	logits := G.NewMatrix(g, tensor.Float32, G.WithShape(rows, classes), G.WithName("logits"))
	labels := G.NewVector(g, tensor.Int32, G.WithShape(rows), G.WithName("labels"))
	loss, err := G.MPSCrossEntropy(logits, labels)
	must(err)
	if _, err := G.Grad(loss, logits); err != nil {
		must(err)
	}
	machine := G.NewTapeMachine(g, G.BindDualValues(logits))
	defer machine.Close()
	must(G.Let(logits, tensor.New(tensor.WithShape(rows, classes), tensor.WithBacking(append([]float32(nil), logitsData...)))))
	must(G.Let(labels, tensor.New(tensor.WithShape(rows), tensor.WithBacking(append([]int32(nil), labelsData...)))))
	must(machine.RunAll())
	grad, err := logits.Grad()
	must(err)
	return append([]float32(nil), grad.(*tensor.Dense).Data().([]float32)...)
}

func (m *mpsMLP) toMLP() *mnist.MLP {
	model := &mnist.MLP{
		InputSize:  m.inputSize,
		HiddenSize: m.hiddenSize,
		Classes:    m.classes,
		W1:         make([]float64, len(m.w1)),
		B1:         make([]float64, len(m.b1)),
		W2:         make([]float64, len(m.w2)),
		B2:         make([]float64, len(m.b2)),
	}
	for i, v := range m.w1 {
		model.W1[i] = float64(v)
	}
	for i, v := range m.b1 {
		model.B1[i] = float64(v)
	}
	for i, v := range m.w2 {
		model.W2[i] = float64(v)
	}
	for i, v := range m.b2 {
		model.B2[i] = float64(v)
	}
	return model
}

func countCorrect(logits []float32, labels []int32, classes int) int {
	var correct int
	for row, label := range labels {
		best := 0
		for classCol := 1; classCol < classes; classCol++ {
			if logits[row*classes+classCol] > logits[row*classes+best] {
				best = classCol
			}
		}
		if best == int(label) {
			correct++
		}
	}
	return correct
}
