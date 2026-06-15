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
	probs, err := G.SoftMax(logits)
	must(err)

	m := G.NewTapeMachine(g)
	defer m.Close()

	must(G.Let(logits, tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{
		1, 2, 3,
		1, 1, 1,
	}))))
	must(m.RunAll())

	result, ok := probs.Value().(*tensor.Dense)
	if !ok {
		log.Fatalf("unexpected result type %T", probs.Value())
	}
	fmt.Printf("%.6v\n", result.Data())
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
