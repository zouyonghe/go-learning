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
	g := G.NewGraph()
	x := G.NewMatrix(g, tensor.Float32, G.WithShape(2, 3), G.WithName("x"))
	w1 := G.NewMatrix(g, tensor.Float32, G.WithShape(3, 4), G.WithName("w1"))
	b1 := G.NewVector(g, tensor.Float32, G.WithShape(4), G.WithName("b1"))
	w2 := G.NewMatrix(g, tensor.Float32, G.WithShape(4, 2), G.WithName("w2"))
	b2 := G.NewVector(g, tensor.Float32, G.WithShape(2), G.WithName("b2"))
	labels := G.NewVector(g, tensor.Int32, G.WithShape(2), G.WithName("labels"))

	l1, err := G.Mul(x, w1)
	must(err)
	l1Bias, err := G.BroadcastAdd(l1, b1, nil, []byte{0})
	must(err)
	h, err := G.MPSReLU(l1Bias)
	must(err)
	logits, err := G.Mul(h, w2)
	must(err)
	logitsBias, err := G.BroadcastAdd(logits, b2, nil, []byte{0})
	must(err)
	loss, err := G.MPSCrossEntropy(logitsBias, labels)
	must(err)

	m := G.NewTapeMachine(g)
	defer m.Close()

	must(G.Let(x, tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{
		1, 2, 3,
		4, 5, 6,
	}))))
	must(G.Let(w1, tensor.New(tensor.WithShape(3, 4), tensor.WithBacking([]float32{
		1, -1, 2, 0,
		0, 3, -2, 1,
		2, 1, 0, -3,
	}))))
	must(G.Let(b1, tensor.New(tensor.WithShape(4), tensor.WithBacking([]float32{1, -2, 0.5, 3}))))
	must(G.Let(w2, tensor.New(tensor.WithShape(4, 2), tensor.WithBacking([]float32{
		1, 2,
		-1, 1,
		0.5, -0.5,
		2, 0,
	}))))
	must(G.Let(b2, tensor.New(tensor.WithShape(2), tensor.WithBacking([]float32{0.25, -0.75}))))
	must(G.Let(labels, tensor.New(tensor.WithShape(2), tensor.WithBacking([]int32{1, 0}))))
	must(m.RunAll())

	result, ok := loss.Value().(*G.F32)
	if !ok {
		log.Fatalf("unexpected loss result type %T", loss.Value())
	}
	fmt.Printf("%.6f\n", result.Data())
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
