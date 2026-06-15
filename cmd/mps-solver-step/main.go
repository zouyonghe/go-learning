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
	labelsData := []int32{1, 0}

	g, loss, nodes := buildGraph()
	if _, err := G.Grad(loss, nodes.w1, nodes.b1, nodes.w2, nodes.b2); err != nil {
		must(err)
	}
	m := G.NewTapeMachine(g, G.BindDualValues(nodes.w1, nodes.b1, nodes.w2, nodes.b2))
	defer m.Close()
	letBatch(nodes, xData, w1Data, b1Data, w2Data, b2Data, labelsData)
	must(m.RunAll())
	before := loss.Value().(*G.F32).Data().(float32)

	solver := G.NewAdamSolver(G.WithLearnRate(0.01))
	must(solver.Step([]G.ValueGrad{nodes.w1, nodes.b1, nodes.w2, nodes.b2}))
	m.Reset()
	letBatch(nodes, xData,
		nodes.w1.Value().(*tensor.Dense).Data().([]float32),
		nodes.b1.Value().(*tensor.Dense).Data().([]float32),
		nodes.w2.Value().(*tensor.Dense).Data().([]float32),
		nodes.b2.Value().(*tensor.Dense).Data().([]float32),
		labelsData)
	must(m.RunAll())
	after := loss.Value().(*G.F32).Data().(float32)
	fmt.Printf("before=%.6f after=%.6f\n", before, after)
}

type graphNodes struct {
	x      *G.Node
	w1     *G.Node
	b1     *G.Node
	w2     *G.Node
	b2     *G.Node
	labels *G.Node
}

func buildGraph() (*G.ExprGraph, *G.Node, graphNodes) {
	g := G.NewGraph()
	nodes := graphNodes{
		x:      G.NewMatrix(g, tensor.Float32, G.WithShape(2, 3), G.WithName("x")),
		w1:     G.NewMatrix(g, tensor.Float32, G.WithShape(3, 4), G.WithName("w1")),
		b1:     G.NewVector(g, tensor.Float32, G.WithShape(4), G.WithName("b1")),
		w2:     G.NewMatrix(g, tensor.Float32, G.WithShape(4, 2), G.WithName("w2")),
		b2:     G.NewVector(g, tensor.Float32, G.WithShape(2), G.WithName("b2")),
		labels: G.NewVector(g, tensor.Int32, G.WithShape(2), G.WithName("labels")),
	}
	l1, err := G.Mul(nodes.x, nodes.w1)
	must(err)
	l1Bias, err := G.BroadcastAdd(l1, nodes.b1, nil, []byte{0})
	must(err)
	hidden, err := G.Rectify(l1Bias)
	must(err)
	logits, err := G.Mul(hidden, nodes.w2)
	must(err)
	logitsBias, err := G.BroadcastAdd(logits, nodes.b2, nil, []byte{0})
	must(err)
	loss, err := G.MPSCrossEntropy(logitsBias, nodes.labels)
	must(err)
	return g, loss, nodes
}

func letBatch(nodes graphNodes, xData, w1Data, b1Data, w2Data, b2Data []float32, labelsData []int32) {
	must(G.Let(nodes.x, tensor.New(tensor.WithShape(2, 3), tensor.WithBacking(append([]float32(nil), xData...)))))
	must(G.Let(nodes.w1, tensor.New(tensor.WithShape(3, 4), tensor.WithBacking(append([]float32(nil), w1Data...)))))
	must(G.Let(nodes.b1, tensor.New(tensor.WithShape(4), tensor.WithBacking(append([]float32(nil), b1Data...)))))
	must(G.Let(nodes.w2, tensor.New(tensor.WithShape(4, 2), tensor.WithBacking(append([]float32(nil), w2Data...)))))
	must(G.Let(nodes.b2, tensor.New(tensor.WithShape(2), tensor.WithBacking(append([]float32(nil), b2Data...)))))
	must(G.Let(nodes.labels, tensor.New(tensor.WithShape(2), tensor.WithBacking(append([]int32(nil), labelsData...)))))
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
