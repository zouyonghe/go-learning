//go:build mps && darwin
// +build mps,darwin

package main

import (
	"fmt"
	"log"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	xData := []float32{
		1, 2, 3,
		4, 5, 6,
	}
	w1Data := []float32{
		1, -1, 2, 0,
		0, 3, -2, 1,
		2, 1, 0, -3,
	}
	b1Data := []float32{1, -2, 0.5, 3}
	w2Data := []float32{
		1, 2,
		-1, 1,
		0.5, -0.5,
		2, 0,
	}
	b2Data := []float32{0.25, -0.75}
	labels := []int32{1, 0}

	before, preRelu, hidden, dLogits := forward(xData, w1Data, b1Data, w2Data, b2Data, labels)

	dHidden := make([]float32, 2*4)
	for row := 0; row < 2; row++ {
		for hiddenCol := 0; hiddenCol < 4; hiddenCol++ {
			var grad float32
			for classCol := 0; classCol < 2; classCol++ {
				grad += dLogits[row*2+classCol] * w2Data[hiddenCol*2+classCol]
			}
			dHidden[row*4+hiddenCol] = grad
		}
	}

	dPreRelu := make([]float32, 2*4)
	for i := range dHidden {
		if preRelu[i] > 0 {
			dPreRelu[i] = dHidden[i]
		}
	}

	const lr float32 = 0.01
	for hiddenCol := 0; hiddenCol < 4; hiddenCol++ {
		for classCol := 0; classCol < 2; classCol++ {
			var grad float32
			for row := 0; row < 2; row++ {
				grad += hidden[row*4+hiddenCol] * dLogits[row*2+classCol]
			}
			w2Data[hiddenCol*2+classCol] -= lr * grad
		}
	}
	for classCol := 0; classCol < 2; classCol++ {
		var grad float32
		for row := 0; row < 2; row++ {
			grad += dLogits[row*2+classCol]
		}
		b2Data[classCol] -= lr * grad
	}

	for inputCol := 0; inputCol < 3; inputCol++ {
		for hiddenCol := 0; hiddenCol < 4; hiddenCol++ {
			var grad float32
			for row := 0; row < 2; row++ {
				grad += xData[row*3+inputCol] * dPreRelu[row*4+hiddenCol]
			}
			w1Data[inputCol*4+hiddenCol] -= lr * grad
		}
	}
	for hiddenCol := 0; hiddenCol < 4; hiddenCol++ {
		var grad float32
		for row := 0; row < 2; row++ {
			grad += dPreRelu[row*4+hiddenCol]
		}
		b1Data[hiddenCol] -= lr * grad
	}

	after, _, _, _ := forward(xData, w1Data, b1Data, w2Data, b2Data, labels)
	fmt.Printf("before=%.6f after=%.6f\n", before, after)
}

func forward(xData, w1Data, b1Data, w2Data, b2Data []float32, labelsData []int32) (float32, []float32, []float32, []float32) {
	g := G.NewGraph()
	x := G.NewMatrix(g, tensor.Float32, G.WithShape(2, 3), G.WithName("x"))
	w1 := G.NewMatrix(g, tensor.Float32, G.WithShape(3, 4), G.WithName("w1"))
	b1 := G.NewVector(g, tensor.Float32, G.WithShape(4), G.WithName("b1"))
	w2 := G.NewMatrix(g, tensor.Float32, G.WithShape(4, 2), G.WithName("w2"))
	b2 := G.NewVector(g, tensor.Float32, G.WithShape(2), G.WithName("b2"))
	labels := G.NewVector(g, tensor.Int32, G.WithShape(2), G.WithName("labels"))

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
	loss, err := G.MPSCrossEntropy(logitsBias, labels)
	must(err)

	m := G.NewTapeMachine(g)
	defer m.Close()
	must(G.Let(x, tensor.New(tensor.WithShape(2, 3), tensor.WithBacking(append([]float32(nil), xData...)))))
	must(G.Let(w1, tensor.New(tensor.WithShape(3, 4), tensor.WithBacking(append([]float32(nil), w1Data...)))))
	must(G.Let(b1, tensor.New(tensor.WithShape(4), tensor.WithBacking(append([]float32(nil), b1Data...)))))
	must(G.Let(w2, tensor.New(tensor.WithShape(4, 2), tensor.WithBacking(append([]float32(nil), w2Data...)))))
	must(G.Let(b2, tensor.New(tensor.WithShape(2), tensor.WithBacking(append([]float32(nil), b2Data...)))))
	must(G.Let(labels, tensor.New(tensor.WithShape(2), tensor.WithBacking(append([]int32(nil), labelsData...)))))
	must(m.RunAll())

	lossValue := loss.Value().(*G.F32).Data().(float32)
	preReluValue := preRelu.Value().(*tensor.Dense).Data().([]float32)
	hiddenValue := hidden.Value().(*tensor.Dense).Data().([]float32)
	logitsValue := logitsBias.Value().(*tensor.Dense).Data().([]float32)
	return lossValue,
		append([]float32(nil), preReluValue...),
		append([]float32(nil), hiddenValue...),
		crossEntropyLogitsGrad(logitsValue, labelsData, 2, 2)
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
	m := G.NewTapeMachine(g, G.BindDualValues(logits))
	defer m.Close()
	must(G.Let(logits, tensor.New(tensor.WithShape(rows, classes), tensor.WithBacking(append([]float32(nil), logitsData...)))))
	must(G.Let(labels, tensor.New(tensor.WithShape(rows), tensor.WithBacking(append([]int32(nil), labelsData...)))))
	must(m.RunAll())
	grad, err := logits.Grad()
	must(err)
	return append([]float32(nil), grad.(*tensor.Dense).Data().([]float32)...)
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
