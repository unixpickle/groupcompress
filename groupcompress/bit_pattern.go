package groupcompress

import "math"

type BitPattern interface {
	uint8 | uint16 | uint32
}

func ExtractBitPattern[T BitPattern](s *BitString, indices []int) T {
	var result T
	for i, j := range indices {
		bit := s.Get(j)
		if bit {
			result |= 1 << i
		}
	}
	return result
}

// An EntropyCounter computes the bitwise or joint entropy
// of a sequence of bit patterns in a rolling fashion.
type EntropyCounter[T BitPattern] struct {
	singleCounts []int
	jointCounts  []T
	count        int
}

func NewEntropyCounter[T BitPattern](numBits int) *EntropyCounter[T] {
	return &EntropyCounter[T]{
		singleCounts: make([]int, numBits),
		jointCounts:  make([]T, 1<<uint(numBits)),
	}
}

// Reset resets the entropy counter as if it had never seen
// any previous bit patterns.
func (e *EntropyCounter[T]) Reset() {
	for i := range e.jointCounts {
		e.jointCounts[i] = 0
	}
	for i := range e.singleCounts {
		e.singleCounts[i] = 0
	}
	e.count = 0
}

// Add adds the pattern to the tally.
func (e *EntropyCounter[T]) Add(pattern T) {
	for i := range e.singleCounts {
		if pattern&(1<<i) != 0 {
			e.singleCounts[i]++
		}
	}
	e.jointCounts[pattern]++
	e.count++
}

// BitwiseEntropy returns the summed bitwise entropy.
func (e *EntropyCounter[T]) BitwiseEntropy() float64 {
	var entropy float64
	for _, count := range e.singleCounts {
		if count == 0 || count == e.count {
			continue
		}
		probTrue := float64(count) / float64(e.count)
		probFalse := 1 - probTrue
		entropy -= probTrue*math.Log(probTrue) + probFalse*math.Log(probFalse)
	}
	return entropy
}

// JointEntropy returns the entropy of the joint
// distribution over all the bits.
func (e *EntropyCounter[T]) JointEntropy() float64 {
	var entropy float64
	for _, count := range e.jointCounts {
		if count == 0 {
			continue
		}
		prob := float64(count) / float64(e.count)
		entropy -= prob * math.Log(prob)
	}
	return entropy
}
