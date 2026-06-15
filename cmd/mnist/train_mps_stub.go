//go:build !mps || !darwin
// +build !mps !darwin

package main

import "log"

func trainMPS(args []string) {
	log.Fatal("train-mps requires macOS with -tags mps")
}
