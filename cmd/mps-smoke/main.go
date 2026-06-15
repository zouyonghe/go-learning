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
	w := G.NewMatrix(g, tensor.Float32, G.WithShape(3, 2), G.WithName("w"))
	b := G.NewVector(g, tensor.Float32, G.WithShape(2), G.WithName("b"))

	xw, err := G.Mul(x, w)
	must(err)
	out, err := G.BroadcastAdd(xw, b, nil, []byte{0})
	must(err)

	m := G.NewTapeMachine(g)
	defer m.Close()

	must(G.Let(x, tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{
		1, 2, 3,
		4, 5, 6,
	}))))
	must(G.Let(w, tensor.New(tensor.WithShape(3, 2), tensor.WithBacking([]float32{
		1, 2,
		3, 4,
		5, 6,
	}))))
	must(G.Let(b, tensor.New(tensor.WithShape(2), tensor.WithBacking([]float32{10, 20}))))
	must(m.RunAll())

	result, ok := out.Value().(*tensor.Dense)
	if !ok {
		log.Fatalf("unexpected result type %T", out.Value())
	}
	fmt.Printf("%v\n", result.Data())
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
