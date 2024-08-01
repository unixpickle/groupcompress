package groupcompress

import (
	"math"
	"math/bits"
)

// An EntropyCounter computes the bitwise or joint entropy
// of a sequence of bit patterns in a rolling fashion.
type EntropyCounter struct {
	singleCounts []float64
	jointCounts  []float64
	count        float64
}

func NewEntropyCounter(numBits int) *EntropyCounter {
	return &EntropyCounter{
		singleCounts: make([]float64, numBits),
		jointCounts:  make([]float64, 1<<uint(numBits)),
	}
}

// NumBits returns the number of bits.
func (e *EntropyCounter) NumBits() int {
	return len(e.singleCounts)
}

// Reset resets the entropy counter as if it had never seen
// any previous bit patterns.
func (e *EntropyCounter) Reset() {
	for i := range e.jointCounts {
		e.jointCounts[i] = 0
	}
	for i := range e.singleCounts {
		e.singleCounts[i] = 0
	}
	e.count = 0
}

// Add adds the pattern to the tally.
func (e *EntropyCounter) Add(pattern BitPattern) {
	for i := range e.singleCounts {
		if pattern&(1<<i) != 0 {
			e.singleCounts[i]++
		}
	}
	e.jointCounts[pattern]++
	e.count++
}

// BitwiseEntropy returns the summed bitwise entropy.
func (e *EntropyCounter) BitwiseEntropy() float64 {
	var entropy float64
	for _, count := range e.singleCounts {
		if count == 0 || count == e.count {
			continue
		}
		probTrue := count / e.count
		probFalse := 1 - probTrue
		entropy -= probTrue*math.Log(probTrue) + probFalse*math.Log(probFalse)
	}
	return entropy
}

// JointEntropy returns the entropy of the joint
// distribution over all the bits.
func (e *EntropyCounter) JointEntropy() float64 {
	var entropy float64
	for _, count := range e.jointCounts {
		if count == 0 {
			continue
		}
		prob := count / e.count
		entropy -= prob * math.Log(prob)
	}
	return entropy
}

// PermutedBitwiseEntropy permutes the joint values and
// returns the new bitwise entropy.
//
// Future calls to BitwiseEntropy() will return this new
// value.
//
// BitPatternhis can be called multiple times, erasing the previous
// result. Successive permutations will not be composed.
func (e *EntropyCounter) PermutedBitwiseEntropy(perm []BitPattern) float64 {
	for i := range e.singleCounts {
		e.singleCounts[i] = 0
	}
	for i, count := range e.jointCounts {
		outVal := perm[i]
		for j, x := range e.singleCounts {
			if outVal&(1<<uint(j)) != 0 {
				e.singleCounts[j] = x + count
			}
		}
	}
	return e.BitwiseEntropy()
}

// Smooth copies other and readjusts the probabilities
// under the assumption that bits will be randomly and
// independently flipped with the given probability.
func (e *EntropyCounter) Smooth(other *EntropyCounter, flipProb float64) {
	for i := range e.jointCounts {
		e.jointCounts[i] = 0
	}
	for i, count := range other.jointCounts {
		pattern := BitPattern(i)
		for j := range other.jointCounts {
			pattern1 := BitPattern(j)
			numFlipped := bits.OnesCount64(uint64(pattern ^ pattern1))
			weight := math.Pow(flipProb, float64(numFlipped)) *
				math.Pow(1-flipProb, float64(len(e.singleCounts)-numFlipped))
			e.jointCounts[pattern1] += count * weight
		}
	}

	// Recompute bitwise probs
	for i := range e.singleCounts {
		e.singleCounts[i] = 0
	}
	for i, count := range e.jointCounts {
		outVal := BitPattern(i)
		for j, x := range e.singleCounts {
			if outVal&(1<<uint(j)) != 0 {
				e.singleCounts[j] = x + count
			}
		}
	}

	e.count = other.count
}

func (e *EntropyCounter) JointCount(x BitPattern) float64 {
	return e.jointCounts[x]
}
