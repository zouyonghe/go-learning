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
	logits := G.NewMatrix(g, tensor.Float32, G.WithShape(2, 3), G.WithName("logits"))
	logProbs, err := G.LogSoftMax(logits)
	must(err)

	m := G.NewTapeMachine(g)
	defer m.Close()

	must(G.Let(logits, tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{
		1, 2, 3,
		1, 1, 1,
	}))))
	must(m.RunAll())

	result, ok := logProbs.Value().(*tensor.Dense)
	if !ok {
		log.Fatalf("unexpected result type %T", logProbs.Value())
	}
	values := result.Data().([]float32)
	fmt.Printf("[%.6f %.6f %.6f %.6f %.6f %.6f]\n", values[0], values[1], values[2], values[3], values[4], values[5])
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
