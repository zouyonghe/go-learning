package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
)

const (
	imageMagic = 2051
	labelMagic = 2049
	imageSize  = 28 * 28
)

type Dataset struct {
	Images [][]float64
	Labels []int
}

func LoadIDXDataset(dir string, train bool) (Dataset, error) {
	imageName := "train-images-idx3-ubyte"
	labelName := "train-labels-idx1-ubyte"
	if !train {
		imageName = "t10k-images-idx3-ubyte"
		labelName = "t10k-labels-idx1-ubyte"
	}

	images, err := readImages(resolveIDXPath(dir, imageName))
	if err != nil {
		return Dataset{}, err
	}
	labels, err := readLabels(resolveIDXPath(dir, labelName))
	if err != nil {
		return Dataset{}, err
	}
	if len(images) != len(labels) {
		return Dataset{}, fmt.Errorf("image/label count mismatch: %d images, %d labels", len(images), len(labels))
	}
	return Dataset{Images: images, Labels: labels}, nil
}

func LoadDigitImage(path string) ([]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}

	bounds := img.Bounds()
	out := make([]float64, imageSize)
	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {
			sx := bounds.Min.X + int(math.Floor(float64(x)*float64(bounds.Dx())/28.0))
			sy := bounds.Min.Y + int(math.Floor(float64(y)*float64(bounds.Dy())/28.0))
			gray := color.GrayModel.Convert(img.At(sx, sy)).(color.Gray)
			out[y*28+x] = float64(gray.Y) / 255.0
		}
	}
	return out, nil
}

func resolveIDXPath(dir, base string) string {
	plain := filepath.Join(dir, base)
	if _, err := os.Stat(plain); err == nil {
		return plain
	}
	gz := plain + ".gz"
	if _, err := os.Stat(gz); err == nil {
		return gz
	}
	return plain
}

func readImages(path string) ([][]float64, error) {
	r, closeFn, err := openMaybeGzip(path)
	if err != nil {
		return nil, err
	}
	defer closeFn()

	var magic, count, rows, cols uint32
	if err := binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != imageMagic {
		return nil, fmt.Errorf("%s: invalid image magic %d", path, magic)
	}
	for _, ptr := range []*uint32{&count, &rows, &cols} {
		if err := binary.Read(r, binary.BigEndian, ptr); err != nil {
			return nil, err
		}
	}
	if rows != 28 || cols != 28 {
		return nil, fmt.Errorf("%s: expected 28x28 images, got %dx%d", path, rows, cols)
	}

	images := make([][]float64, count)
	buf := make([]byte, imageSize)
	for i := range images {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, err
		}
		img := make([]float64, imageSize)
		for j, v := range buf {
			img[j] = float64(v) / 255.0
		}
		images[i] = img
	}
	return images, nil
}

func readLabels(path string) ([]int, error) {
	r, closeFn, err := openMaybeGzip(path)
	if err != nil {
		return nil, err
	}
	defer closeFn()

	var magic, count uint32
	if err := binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != labelMagic {
		return nil, fmt.Errorf("%s: invalid label magic %d", path, magic)
	}
	if err := binary.Read(r, binary.BigEndian, &count); err != nil {
		return nil, err
	}

	buf := make([]byte, count)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	labels := make([]int, count)
	for i, v := range buf {
		labels[i] = int(v)
	}
	return labels, nil
}

type readCloser interface {
	Read([]byte) (int, error)
	Close() error
}

func openMaybeGzip(path string) (readCloser, func(), error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	if !strings.HasSuffix(path, ".gz") {
		return f, func() { _ = f.Close() }, nil
	}
	gz, err := gzip.NewReader(f)
	if err != nil {
		_ = f.Close()
		return nil, nil, err
	}
	return gz, func() { _ = gz.Close(); _ = f.Close() }, nil
}
