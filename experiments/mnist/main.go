package main

import (
	"encoding/json"
	"flag"
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"math/rand"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/groupcompress/groupcompress"
	"github.com/unixpickle/mnistlite"
)

func main() {
	var batchSize int
	var numBits int
	var samples int
	var sampleGrid int
	var priorSamples int
	var samplePath string
	var savePath string
	flag.IntVar(&batchSize, "batch-size", 128, "examples per layer")
	flag.IntVar(&numBits, "num-bits", 3, "bits per group")
	flag.IntVar(&samples, "samples", 100000, "groups to sample")
	flag.IntVar(&sampleGrid, "sample-grid", 4, "size of sample grid")
	flag.IntVar(&priorSamples, "prior-samples", 60000, "number of samples to compute prior")
	flag.StringVar(&samplePath, "sample-path", "samples.png", "path of samples output image")
	flag.StringVar(&savePath, "save-path", "model.json", "path to save model checkpoint")
	flag.Parse()

	model := []*groupcompress.Transform[uint8]{}
	if _, err := os.Stat(savePath); err == nil {
		log.Printf("Loading checkpoint: %s ...", savePath)
		data, err := os.ReadFile(savePath)
		essentials.Must(err)
		essentials.Must(json.Unmarshal(data, &model))
	}

	batches := LoadBatches(batchSize)
	for {
		batch := <-batches
		for _, example := range batch {
			for _, layer := range model {
				layer.Apply(example)
			}
		}
		initEntropy := groupcompress.MeanBitwiseEntropy(batch)
		result := groupcompress.EntropySearch[uint8](
			batch,
			numBits,
			samples,
		)
		log.Printf("step %d: loss=%f reduction=%f", len(model), initEntropy*28*28, result.EntropyReduction())
		model = append(model, result.Transform)

		prior := EstimatePrior(batches, model, priorSamples)
		var samples []*groupcompress.BitString
		for i := 0; i < sampleGrid*sampleGrid; i++ {
			samples = append(samples, Sample(prior, model))
		}
		WriteGrid(samplePath, samples)

		data, err := json.Marshal(model)
		essentials.Must(err)
		essentials.Must(os.WriteFile(savePath+".tmp", data, 0644))
		essentials.Must(os.Rename(savePath+".tmp", savePath))
	}
}

func EstimatePrior(
	batches <-chan []*groupcompress.BitString,
	model []*groupcompress.Transform[uint8],
	numSamples int,
) []float64 {
	counts := make([]float64, 28*28)

	sampled := 0

OuterLoop:
	for batch := range batches {
		for _, x := range batch {
			for _, layer := range model {
				layer.Apply(x)
			}
			for i := range counts {
				if x.Get(i) {
					counts[i]++
				}
			}
			sampled++
			if sampled == numSamples {
				break OuterLoop
			}
		}
	}

	for i, x := range counts {
		counts[i] = x / float64(numSamples)
	}
	return counts
}

func Sample(prior []float64, layers []*groupcompress.Transform[uint8]) *groupcompress.BitString {
	result := groupcompress.NewBitString(len(prior))
	for i, p := range prior {
		if rand.Float64() < p {
			result.Set(i, true)
		}
	}

	for i := len(layers) - 1; i >= 0; i-- {
		layers[i].Inverse().Apply(result)
	}

	return result
}

func WriteGrid(path string, samples []*groupcompress.BitString) {
	sideLength := int(math.Sqrt(float64(len(samples))))
	if sideLength*sideLength != len(samples) {
		panic("must write a square number of samples")
	}
	img := image.NewGray(image.Rect(0, 0, sideLength*28, sideLength*28))
	for i, sample := range samples {
		x := 28 * (i % sideLength)
		y := 28 * (i / sideLength)
		for subX := 0; subX < 28; subX++ {
			for subY := 0; subY < 28; subY++ {
				bit := sample.Get(subX + subY*28)
				if bit {
					img.SetGray(x+subX, y+subY, color.Gray{Y: uint8(255)})
				} else {
					img.SetGray(x+subX, y+subY, color.Gray{Y: uint8(0)})
				}
			}
		}
	}
	w, err := os.Create(path)
	essentials.Must(err)
	defer w.Close()
	essentials.Must(png.Encode(w, img))
}

func LoadBatches(batchSize int) <-chan []*groupcompress.BitString {
	result := make(chan []*groupcompress.BitString, 1)

	go func() {
		data := mnistlite.LoadTrainingDataSet()
		var batch []*groupcompress.BitString
		for {
			perm := rand.Perm(len(data.Samples))
			for _, i := range perm {
				sample := data.Samples[i]
				bs := groupcompress.NewBitString(len(sample.Intensities))
				for i, prob := range sample.Intensities {
					if rand.Float64() < prob {
						bs.Set(i, true)
					}
				}
				batch = append(batch, bs)
				if len(batch) == batchSize {
					result <- batch
					batch = nil
				}
			}
		}
	}()

	return result
}
