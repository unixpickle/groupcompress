package groupcompress

import (
	"math/rand"
	"runtime"
	"sort"
)

type EntropyDeltas struct {
	OldBitwise float64
	NewBitwise float64
	Joint      float64
}

func (e *EntropyDeltas) Reduction() float64 {
	return e.OldBitwise - e.NewBitwise
}

type EntropySearchResult[T BitPattern] struct {
	Transform *Transform[T]

	// Deltas is the (smoothed) entropy deltas.
	Deltas EntropyDeltas

	// RawDeltas is the entropy deltas on the original
	// unsmoothed data, which were not used for selection.
	RawDeltas EntropyDeltas
}

// EntropySearch finds a transformation that reduces
func EntropySearch[T BitPattern](
	data []*BitString,
	numBits int,
	samples int,
	permSearch PermSearch[T],
	prefix []int,
	flipProb float64,
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
			results <- entropySearch[T](rng, data, numBits, samplesPerWorker, permSearch, prefix,
				flipProb)
		}()
	}
	var best *EntropySearchResult[T]
	for i := 0; i < workers; i++ {
		result := <-results
		if best == nil || result.Deltas.Reduction() > best.Deltas.Reduction() {
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
	prefix []int,
	flipProb float64,
) *EntropySearchResult[T] {
	var best EntropySearchResult[T]

	// Pre-allocate and reuse
	entropyCounter := NewEntropyCounter[T](numBits)
	smoothCounter := NewEntropyCounter[T](numBits)

	for i := 0; i < samples; i++ {
		indices := sampleIndices(rng, data[0].NumBits, numBits, prefix)

		entropyCounter.Reset()
		for _, datum := range data {
			pattern := ExtractBitPattern[T](datum, indices)
			entropyCounter.Add(pattern)
		}
		bitwiseOld := entropyCounter.BitwiseEntropy()
		jointEntropy := entropyCounter.JointEntropy()
		smoothCounter.Smooth(entropyCounter, flipProb)
		smoothBitwiseOld := smoothCounter.BitwiseEntropy()
		smoothJointEntropy := smoothCounter.JointEntropy()

		if i > 0 && smoothBitwiseOld-smoothJointEntropy < best.Deltas.Reduction() {
			continue
		}

		bestPerm := permSearch.Search(rng, 1<<uint(numBits), smoothCounter)
		bestDelta := smoothBitwiseOld - smoothCounter.PermutedBitwiseEntropy(bestPerm)

		if bestDelta >= best.Deltas.Reduction() || i == 0 {
			best = EntropySearchResult[T]{
				Transform: &Transform[T]{
					Indices: indices,
					Mapping: bestPerm,
				},
				Deltas: EntropyDeltas{
					OldBitwise: smoothBitwiseOld,
					NewBitwise: smoothBitwiseOld - bestDelta,
					Joint:      smoothJointEntropy,
				},
				RawDeltas: EntropyDeltas{
					OldBitwise: bitwiseOld,
					NewBitwise: entropyCounter.PermutedBitwiseEntropy(bestPerm),
					Joint:      jointEntropy,
				},
			}
		}
	}
	return &best
}

func sampleIndices(rng *rand.Rand, dim, count int, prefix []int) []int {
	indices := make([]int, 0, count)
	indices = append(indices, prefix...)
	for i := len(prefix); i < count; i++ {
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
