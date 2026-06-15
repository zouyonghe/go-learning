package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadIDXDataset(t *testing.T) {
	dir := t.TempDir()
	writeImages(t, filepath.Join(dir, "train-images-idx3-ubyte.gz"), [][]byte{
		make([]byte, imageSize),
		filledImage(255),
	})
	writeLabels(t, filepath.Join(dir, "train-labels-idx1-ubyte.gz"), []byte{3, 7})

	set, err := LoadIDXDataset(dir, true)
	if err != nil {
		t.Fatalf("LoadIDXDataset() error = %v", err)
	}
	if len(set.Images) != 2 || len(set.Labels) != 2 {
		t.Fatalf("got %d images and %d labels", len(set.Images), len(set.Labels))
	}
	if set.Labels[0] != 3 || set.Labels[1] != 7 {
		t.Fatalf("unexpected labels: %v", set.Labels)
	}
	if set.Images[1][0] != 1.0 {
		t.Fatalf("expected normalized pixel 1.0, got %f", set.Images[1][0])
	}
}

func TestModelPredict(t *testing.T) {
	m := NewMLP(imageSize, 8, 10)
	preds, err := m.Predict(make([]float64, imageSize), 3)
	if err != nil {
		t.Fatalf("Predict() error = %v", err)
	}
	if len(preds) != 3 {
		t.Fatalf("expected 3 predictions, got %d", len(preds))
	}
}

func filledImage(v byte) []byte {
	img := make([]byte, imageSize)
	for i := range img {
		img[i] = v
	}
	return img
}

func writeImages(t *testing.T, path string, images [][]byte) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	gz := gzip.NewWriter(f)
	mustWrite(t, gz, uint32(imageMagic))
	mustWrite(t, gz, uint32(len(images)))
	mustWrite(t, gz, uint32(28))
	mustWrite(t, gz, uint32(28))
	for _, img := range images {
		if len(img) != imageSize {
			t.Fatalf("bad image length %d", len(img))
		}
		if _, err := gz.Write(img); err != nil {
			t.Fatal(err)
		}
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func writeLabels(t *testing.T, path string, labels []byte) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	gz := gzip.NewWriter(f)
	mustWrite(t, gz, uint32(labelMagic))
	mustWrite(t, gz, uint32(len(labels)))
	if _, err := gz.Write(labels); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func mustWrite(t *testing.T, f interface{ Write([]byte) (int, error) }, v uint32) {
	t.Helper()
	if err := binary.Write(f, binary.BigEndian, v); err != nil {
		t.Fatal(err)
	}
}
