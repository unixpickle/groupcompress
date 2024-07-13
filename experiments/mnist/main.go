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

type Model[T groupcompress.BitPattern] groupcompress.TransformList[T]

func (m Model[T]) Inverse() Model[T] {
	return Model[T](groupcompress.TransformList[T](m).Inverse())
}

func (m Model[T]) Apply(b *groupcompress.BitString) {
	groupcompress.TransformList[T](m).Apply(b)
}

type Args[T groupcompress.BitPattern] struct {
	SavePath string

	// Hyperparameters
	BatchSize    int
	NumBits      int
	BitChunkSize int
	Samples      int
	SearchType   string
	EvoSearch    groupcompress.EvoPermSearch[T]

	// Sampling
	SampleGrid   int
	PriorSamples int
	SamplePath   string
}

func (a *Args[T]) AddToFlags(fs *flag.FlagSet) {
	fs.IntVar(&a.BatchSize, "batch-size", 128, "examples per layer")
	fs.IntVar(&a.NumBits, "num-bits", 3, "bits per group")
	fs.IntVar(&a.BitChunkSize, "bit-chunk-size", 0, "bits to greedily search at a time")
	fs.IntVar(&a.Samples, "samples", 100000, "groups to sample")
	fs.StringVar(&a.SearchType, "search-type", "evo", "options: evo, greedybit, singlebit")
	fs.IntVar(&a.EvoSearch.Generations, "perm-generations", 0,
		"generations of evolutionary permutation search")
	fs.IntVar(&a.EvoSearch.Population, "perm-population", 1000,
		"population of evolutionary permutation search")
	fs.IntVar(&a.EvoSearch.Mutations, "perm-mutations", 0,
		"mutations per example in evolutionary permutation search")
	fs.IntVar(&a.EvoSearch.MaxSwapsPerMutation, "perm-swaps", 5,
		"maximum swaps per mutation in evolutionary permutation search")
	fs.IntVar(&a.SampleGrid, "sample-grid", 4, "size of sample grid")
	fs.IntVar(&a.PriorSamples, "prior-samples", 60000, "number of samples to compute prior")
	fs.StringVar(&a.SamplePath, "sample-path", "samples.png", "path of samples output image")
	fs.StringVar(&a.SavePath, "save-path", "model.json", "path to save model checkpoint")
}

func main() {
	// Attempt to figure out bit size to use.
	fs := flag.NewFlagSet("mnist", flag.ContinueOnError)
	dummyArgs := &Args[uint8]{}
	dummyArgs.AddToFlags(fs)
	fs.Parse(os.Args[1:])
	if dummyArgs.NumBits <= 8 {
		Main[uint8]()
	} else if dummyArgs.NumBits <= 16 {
		Main[uint16]()
	} else if dummyArgs.NumBits <= 32 {
		Main[uint32]()
	} else {
		essentials.Die("cannot operate on > 32 bits.")
	}
}

func Main[T groupcompress.BitPattern]() {
	var a Args[T]
	a.AddToFlags(flag.CommandLine)
	flag.Parse()

	var permSearch groupcompress.PermSearch[T]
	if a.SearchType == "evo" {
		permSearch = &a.EvoSearch
	} else if a.SearchType == "greedybit" {
		permSearch = &groupcompress.GreedyBitwiseSearch[T]{}
	} else if a.SearchType == "singlebit" {
		permSearch = &groupcompress.SingleBitPartitionSearch[T]{}
	} else {
		essentials.Die("unsupported search type:", a.SearchType)
	}

	var model Model[T]

	if _, err := os.Stat(a.SavePath); err == nil {
		log.Printf("Loading checkpoint: %s ...", a.SavePath)
		data, err := os.ReadFile(a.SavePath)
		essentials.Must(err)
		essentials.Must(json.Unmarshal(data, &model))
	}

	batches := LoadBatches(a.BatchSize)
	for {
		batch := <-batches
		for _, example := range batch {
			for _, layer := range model {
				layer.Apply(example)
			}
		}
		initEntropy := groupcompress.MeanBitwiseEntropy(batch)
		var result *groupcompress.EntropySearchResult[T]
		if a.BitChunkSize == 0 {
			result = groupcompress.EntropySearch[T](
				batch,
				a.NumBits,
				a.Samples,
				permSearch,
				nil,
			)
		} else {
			var prefix []int
			for len(prefix) < a.NumBits {
				targetNumBits := essentials.MinInt(a.NumBits, len(prefix)+a.BitChunkSize)
				result = groupcompress.EntropySearch[T](
					batch,
					targetNumBits,
					a.Samples,
					permSearch,
					prefix,
				)
				prefix = result.Transform.Indices
			}
		}
		log.Printf("step %d: loss=%f reduction=%f", len(model), initEntropy*28*28, result.EntropyReduction())
		model = append(model, result.Transform)

		prior := EstimatePrior(batches, model, a.PriorSamples)
		var samples []*groupcompress.BitString
		for i := 0; i < a.SampleGrid*a.SampleGrid; i++ {
			samples = append(samples, Sample(prior, model))
		}
		WriteGrid(a.SamplePath, samples)

		data, err := json.Marshal(model)
		essentials.Must(err)
		essentials.Must(os.WriteFile(a.SavePath+".tmp", data, 0644))
		essentials.Must(os.Rename(a.SavePath+".tmp", a.SavePath))
	}
}

func EstimatePrior[T groupcompress.BitPattern](
	batches <-chan []*groupcompress.BitString,
	model Model[T],
	numSamples int,
) []float64 {
	counts := make([]float64, 28*28)

	sampled := 0

OuterLoop:
	for batch := range batches {
		for _, x := range batch {
			model.Apply(x)
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

func Sample[T groupcompress.BitPattern](prior []float64, model Model[T]) *groupcompress.BitString {
	result := groupcompress.NewBitString(len(prior))
	for i, p := range prior {
		if rand.Float64() < p {
			result.Set(i, true)
		}
	}
	model.Inverse().Apply(result)
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
