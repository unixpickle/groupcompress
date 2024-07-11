package groupcompress

import (
	"math/rand"
	"runtime"
	"sort"
)

type EntropySearchResult[T BitPattern] struct {
	Transform         *Transform[T]
	OldBitwiseEntropy float64
	NewBitwiseEntropy float64
	JointEntropy      float64
}

func (e *EntropySearchResult[T]) EntropyReduction() float64 {
	return e.OldBitwiseEntropy - e.NewBitwiseEntropy
}

// EntropySearch finds a transformation that reduces
func EntropySearch[T BitPattern](
	data []*BitString,
	numBits int,
	samples int,
	permSearch PermSearch[T],
) *EntropySearchResult[T] {
	workers := runtime.GOMAXPROCS(0)
	if workers > samples {
		workers = samples
	}
	nExtra := samples % workers
	results := make(chan *EntropySearchResult[T], workers)
	for i := 0; i < workers; i++ {
		samplesPerWorker := samples / workers
		if i < nExtra {
			samplesPerWorker += 1
		}
		rng := rand.New(rand.NewSource(rand.Int63()))
		go func() {
			results <- entropySearch[T](rng, data, numBits, samplesPerWorker, permSearch)
		}()
	}
	var best *EntropySearchResult[T]
	for i := 0; i < workers; i++ {
		result := <-results
		if best == nil || result.EntropyReduction() > best.EntropyReduction() {
			best = result
		}
	}
	return best
}

func entropySearch[T BitPattern](
	rng *rand.Rand,
	data []*BitString,
	numBits int,
	samples int,
	permSearch PermSearch[T],
) *EntropySearchResult[T] {
	var best EntropySearchResult[T]

	// Pre-allocate and reuse
	entropyCounter := NewEntropyCounter[T](numBits)

	for i := 0; i < samples; i++ {
		indices := sampleIndices(rng, data[0].NumBits, numBits)

		entropyCounter.Reset()
		for _, datum := range data {
			pattern := ExtractBitPattern[T](datum, indices)
			entropyCounter.Add(pattern)
		}
		bitwiseOld := entropyCounter.BitwiseEntropy()
		jointEntropy := entropyCounter.JointEntropy()
		if i > 0 && bitwiseOld-jointEntropy < best.EntropyReduction() {
			continue
		}

		bestPerm := permSearch.Search(rng, 1<<uint(numBits), entropyCounter)
		bestDelta := bitwiseOld - entropyCounter.PermutedBitwiseEntropy(bestPerm)

		if bestDelta >= best.EntropyReduction() || i == 0 {
			best = EntropySearchResult[T]{
				Transform: &Transform[T]{
					Indices: indices,
					Mapping: bestPerm,
				},
				OldBitwiseEntropy: bitwiseOld,
				NewBitwiseEntropy: bitwiseOld - bestDelta,
				JointEntropy:      jointEntropy,
			}
		}
	}
	return &best
}

func sampleIndices(rng *rand.Rand, dim, count int) []int {
	indices := make([]int, 0, count)
	for i := 0; i < count; i++ {
		n := rng.Intn(dim - i)
		for _, j := range indices {
			if n >= j {
				n += 1
			}
		}
		indices = append(indices, n)
		sort.Ints(indices)
	}
	return indices
}
