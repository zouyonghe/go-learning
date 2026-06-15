package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"go-learning/internal/mnist"
)

func main() {
	log.SetFlags(0)

	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}

	switch os.Args[1] {
	case "train":
		train(os.Args[2:])
	case "train-mps":
		trainMPS(os.Args[2:])
	case "infer":
		infer(os.Args[2:])
	case "help", "-h", "--help":
		usage()
	default:
		log.Fatalf("unknown command %q\n\nRun: go run ./cmd/mnist help", os.Args[1])
	}
}

func train(args []string) {
	fs := flag.NewFlagSet("train", flag.ExitOnError)
	dataDir := fs.String("data", "./data", "directory containing MNIST IDX files")
	epochs := fs.Int("epochs", 5, "number of training epochs")
	batchSize := fs.Int("batch", 64, "mini-batch size")
	lr := fs.Float64("lr", 0.01, "learning rate")
	modelPath := fs.String("model", "./mnist.gob", "checkpoint path")
	hidden := fs.Int("hidden", 128, "hidden layer size")
	limit := fs.Int("limit", 0, "optional max training samples, 0 means all")
	must(fs.Parse(args))

	set, err := mnist.LoadIDXDataset(*dataDir, true)
	must(err)
	if *limit > 0 && *limit < len(set.Images) {
		set.Images = set.Images[:*limit]
		set.Labels = set.Labels[:*limit]
	}

	model := mnist.NewMLP(28*28, *hidden, 10)
	stats, err := model.Train(set, mnist.TrainConfig{
		Epochs:    *epochs,
		BatchSize: *batchSize,
		LR:        *lr,
	})
	must(err)

	for _, stat := range stats {
		fmt.Printf("epoch=%d loss=%.4f accuracy=%.2f%%\n", stat.Epoch, stat.Loss, stat.Accuracy*100)
	}

	must(os.MkdirAll(filepath.Dir(*modelPath), 0o755))
	must(model.Save(*modelPath))
	fmt.Printf("saved model to %s\n", *modelPath)
}

func infer(args []string) {
	fs := flag.NewFlagSet("infer", flag.ExitOnError)
	modelPath := fs.String("model", "./mnist.gob", "checkpoint path")
	imagePath := fs.String("image", "", "PNG/JPEG digit image path")
	topK := fs.Int("topk", 3, "number of predictions to print")
	must(fs.Parse(args))

	if *imagePath == "" {
		log.Fatal("missing -image")
	}

	model, err := mnist.LoadModel(*modelPath)
	must(err)

	input, err := mnist.LoadDigitImage(*imagePath)
	must(err)

	predictions, err := model.Predict(input, *topK)
	must(err)
	for _, pred := range predictions {
		fmt.Printf("digit=%d probability=%.4f\n", pred.Label, pred.Probability)
	}
}

func usage() {
	fmt.Println(`MNIST digit recognizer in Go.

Commands:
  train   Train a small MLP on MNIST IDX files
  train-mps
          Train a small MLP with the experimental MPSGraph path
  infer   Run inference on a PNG/JPEG digit image

Examples:
  go run ./cmd/mnist train -data ./data -epochs 5 -batch 64 -lr 0.01 -model ./mnist.gob
  ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go run -tags mps ./cmd/mnist train-mps -data ./data -epochs 1 -batch 32 -limit 1000 -model ./mnist.gob
  go run ./cmd/mnist infer -model ./mnist.gob -image ./digit.png`)
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
