//go:build !mps || !darwin
// +build !mps !darwin

package main

import "log"

func trainMPSSolver(args []string) {
	log.Fatal("train-mps-solver requires macOS with -tags mps")
}
