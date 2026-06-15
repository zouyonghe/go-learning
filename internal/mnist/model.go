package mnist

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"time"

	"gorgonia.org/tensor"
)

type MLP struct {
	InputSize  int
	HiddenSize int
	Classes    int
	W1         []float64
	B1         []float64
	W2         []float64
	B2         []float64
}

type TrainConfig struct {
	Epochs    int
	BatchSize int
	LR        float64
}

type EpochStat struct {
	Epoch    int
	Loss     float64
	Accuracy float64
}

type Prediction struct {
	Label       int
	Probability float64
}

func NewMLP(inputSize, hiddenSize, classes int) *MLP {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	m := &MLP{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		Classes:    classes,
		W1:         make([]float64, inputSize*hiddenSize),
		B1:         make([]float64, hiddenSize),
		W2:         make([]float64, hiddenSize*classes),
		B2:         make([]float64, classes),
	}
	for i := range m.W1 {
		m.W1[i] = rng.NormFloat64() * math.Sqrt(2.0/float64(inputSize))
	}
	for i := range m.W2 {
		m.W2[i] = rng.NormFloat64() * math.Sqrt(2.0/float64(hiddenSize))
	}
	return m
}

func (m *MLP) Train(data Dataset, cfg TrainConfig) ([]EpochStat, error) {
	if cfg.Epochs <= 0 {
		return nil, fmt.Errorf("epochs must be positive")
	}
	if cfg.BatchSize <= 0 {
		return nil, fmt.Errorf("batch size must be positive")
	}
	if cfg.LR <= 0 {
		return nil, fmt.Errorf("learning rate must be positive")
	}
	if len(data.Images) == 0 {
		return nil, fmt.Errorf("empty dataset")
	}

	stats := make([]EpochStat, 0, cfg.Epochs)
	indices := rand.Perm(len(data.Images))
	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		rand.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })

		var totalLoss float64
		var correct int
		for start := 0; start < len(indices); start += cfg.BatchSize {
			end := start + cfg.BatchSize
			if end > len(indices) {
				end = len(indices)
			}
			loss, batchCorrect := m.trainBatch(data, indices[start:end], cfg.LR)
			totalLoss += loss * float64(end-start)
			correct += batchCorrect
		}

		stats = append(stats, EpochStat{
			Epoch:    epoch,
			Loss:     totalLoss / float64(len(data.Images)),
			Accuracy: float64(correct) / float64(len(data.Images)),
		})
	}
	return stats, nil
}

func (m *MLP) Predict(input []float64, topK int) ([]Prediction, error) {
	if len(input) != m.InputSize {
		return nil, fmt.Errorf("expected input size %d, got %d", m.InputSize, len(input))
	}
	if topK <= 0 || topK > m.Classes {
		topK = m.Classes
	}
	_, probabilities := m.forward(input)
	preds := make([]Prediction, m.Classes)
	for i, p := range probabilities {
		preds[i] = Prediction{Label: i, Probability: p}
	}
	sort.Slice(preds, func(i, j int) bool { return preds[i].Probability > preds[j].Probability })
	return preds[:topK], nil
}

func (m *MLP) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(m)
}

func LoadModel(path string) (*MLP, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var m MLP
	if err := gob.NewDecoder(f).Decode(&m); err != nil {
		return nil, err
	}
	return &m, nil
}

func (m *MLP) trainBatch(data Dataset, batch []int, lr float64) (float64, int) {
	dW1 := make([]float64, len(m.W1))
	dB1 := make([]float64, len(m.B1))
	dW2 := make([]float64, len(m.W2))
	dB2 := make([]float64, len(m.B2))

	var totalLoss float64
	var correct int
	for _, idx := range batch {
		x := data.Images[idx]
		y := data.Labels[idx]
		hidden, probs := m.forward(x)
		if argmax(probs) == y {
			correct++
		}
		totalLoss += -math.Log(math.Max(probs[y], 1e-12))

		dz2 := append([]float64(nil), probs...)
		dz2[y] -= 1
		for h := 0; h < m.HiddenSize; h++ {
			for c := 0; c < m.Classes; c++ {
				dW2[h*m.Classes+c] += hidden[h] * dz2[c]
			}
		}
		for c := 0; c < m.Classes; c++ {
			dB2[c] += dz2[c]
		}

		dh := make([]float64, m.HiddenSize)
		for h := 0; h < m.HiddenSize; h++ {
			for c := 0; c < m.Classes; c++ {
				dh[h] += m.W2[h*m.Classes+c] * dz2[c]
			}
			if hidden[h] <= 0 {
				dh[h] = 0
			}
		}

		for i, xv := range x {
			for h := 0; h < m.HiddenSize; h++ {
				dW1[i*m.HiddenSize+h] += xv * dh[h]
			}
		}
		for h := 0; h < m.HiddenSize; h++ {
			dB1[h] += dh[h]
		}
	}

	scale := lr / float64(len(batch))
	for i := range m.W1 {
		m.W1[i] -= scale * dW1[i]
	}
	for i := range m.B1 {
		m.B1[i] -= scale * dB1[i]
	}
	for i := range m.W2 {
		m.W2[i] -= scale * dW2[i]
	}
	for i := range m.B2 {
		m.B2[i] -= scale * dB2[i]
	}
	return totalLoss / float64(len(batch)), correct
}

func (m *MLP) forward(input []float64) ([]float64, []float64) {
	// Keep tensor construction in the hot path lightweight while making the model state usable with Gorgonia's tensor package.
	_ = tensor.New(tensor.WithShape(1, m.InputSize), tensor.WithBacking(input))

	hidden := make([]float64, m.HiddenSize)
	for h := 0; h < m.HiddenSize; h++ {
		sum := m.B1[h]
		for i, x := range input {
			sum += x * m.W1[i*m.HiddenSize+h]
		}
		if sum > 0 {
			hidden[h] = sum
		}
	}

	logits := make([]float64, m.Classes)
	for c := 0; c < m.Classes; c++ {
		sum := m.B2[c]
		for h, v := range hidden {
			sum += v * m.W2[h*m.Classes+c]
		}
		logits[c] = sum
	}
	return hidden, softmax(logits)
}

func softmax(logits []float64) []float64 {
	maxLogit := logits[0]
	for _, v := range logits[1:] {
		if v > maxLogit {
			maxLogit = v
		}
	}
	var sum float64
	out := make([]float64, len(logits))
	for i, v := range logits {
		out[i] = math.Exp(v - maxLogit)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func argmax(values []float64) int {
	best := 0
	for i := 1; i < len(values); i++ {
		if values[i] > values[best] {
			best = i
		}
	}
	return best
}
