package groupcompress

import (
	"math"
	"math/rand"
	"sort"
)

// A PermSearch implements an algorithm for optimizing an
// objective over the space of permutations.
type PermSearch[T BitPattern] interface {
	// Search finds a permutation to maximize f.
	// The size argument indicates the number of elements
	// in the permutation.
	// The rng may be used as part of the search.
	Search(rng *rand.Rand, size int, e *EntropyCounter[T]) []T
}

// RandomPermSearch is a PermSearch which samples random
// permutations and selects the best one.
type RandomPermSearch[T BitPattern] struct {
	Samples int
}

// Search tries r.Samples permutations and returns the one
// with the best objective value.
func (r *RandomPermSearch[T]) Search(rng *rand.Rand, size int, e *EntropyCounter[T]) []T {
	var bestPerm []T
	var bestObj float64
	for i := 0; i < r.Samples; i++ {
		perm := make([]T, size)
		for i, x := range rng.Perm(len(perm)) {
			perm[i] = T(x)
		}
		obj := -e.PermutedBitwiseEntropy(perm)
		if obj > bestObj || i == 0 {
			bestPerm = perm
			bestObj = obj
		}
	}
	return bestPerm
}

// EvoPermSearch is a PermSearch which uses an evolutionary
// algorithm to gradually improve a set of permutations.
type EvoPermSearch[T BitPattern] struct {
	// Generations is the number of rounds to run the
	// algorithm for.
	Generations int

	// Population is the number of high-objective samples
	// to mutate at each generation.
	Population int

	// Mutations is the number of children each sample in
	// the population will be mutated to produce.
	Mutations int

	// MaxSwapsPerMutation determines how aggressive
	// mutations are. If it is 1, each mutation is a single
	// swap between elements. If it is N, at most N swaps
	// will be performed.
	MaxSwapsPerMutation int
}

// Search performs evolutionary search.
func (e *EvoPermSearch[T]) Search(rng *rand.Rand, size int, ec *EntropyCounter[T]) []T {
	if e.Generations == 0 {
		return (&RandomPermSearch[T]{
			Samples: e.Population * (1 + e.Mutations),
		}).Search(rng, size, ec)
	}

	population := make(evoPopulation[T], e.Population*(1+e.Mutations))
	for i := range population {
		var p evoSample[T]
		p.Perm = make([]T, size)
		if i == 0 {
			for j := range p.Perm {
				p.Perm[j] = T(j)
			}
		} else {
			for j, x := range rng.Perm(size) {
				p.Perm[j] = T(x)
			}
		}
		p.Value = -ec.PermutedBitwiseEntropy(p.Perm)
		population[i] = p
	}
	sort.Sort(population)

	for i := 0; i < e.Generations; i++ {
		for j := 0; j < e.Population; j++ {
			parent := population[j]
			for k := 0; k < e.Mutations; k++ {
				childIdx := e.Population + j*e.Mutations + k
				child := &population[childIdx]
				copy(child.Perm, parent.Perm)
				numMutations := rng.Intn(e.MaxSwapsPerMutation) + 1
				for l := 0; l < numMutations; l++ {
					i1 := rand.Intn(size)
					i2 := rand.Intn(size - 1)
					if i2 >= i1 {
						i2++
					}
					child.Perm[i1], child.Perm[i2] = child.Perm[i2], child.Perm[i1]
				}
				child.Value = -ec.PermutedBitwiseEntropy(child.Perm)
			}
		}
		sort.Sort(population)
	}
	return population[0].Perm
}

type evoSample[T BitPattern] struct {
	Perm  []T
	Value float64
}

type evoPopulation[T BitPattern] []evoSample[T]

func (e evoPopulation[T]) Len() int {
	return len(e)
}

func (e evoPopulation[T]) Less(i, j int) bool {
	return e[i].Value > e[j].Value
}

func (e evoPopulation[T]) Swap(i, j int) {
	e[i], e[j] = e[j], e[i]
}

type GreedyBitwiseSearch[T BitPattern] struct {
	MinDifference float64
}

func (g *GreedyBitwiseSearch[T]) Search(rng *rand.Rand, size int, e *EntropyCounter[T]) []T {
	perm := make([]T, size)
	for i := range perm {
		perm[i] = T(i)
	}
	loss := -e.PermutedBitwiseEntropy(perm)
	inverse := append([]T{}, perm...)
	for bitMask := uint64(1); (1 << bitMask) < size; bitMask <<= 1 {
		for i, t := range perm {
			other := t ^ T(bitMask)
			otherIdx := inverse[other]
			if math.Abs(e.JointCount(t)-e.JointCount(other)) < g.MinDifference {
				continue
			}
			perm[i], perm[otherIdx] = other, t
			newLoss := -e.PermutedBitwiseEntropy(perm)
			if newLoss > loss {
				loss = newLoss
				inverse[t], inverse[other] = otherIdx, T(i)
			} else {
				perm[i], perm[otherIdx] = t, other
			}
		}
	}
	return perm
}

type SingleBitPartitionSearch[T BitPattern] struct {
	MinDifference float64
}

func (g *SingleBitPartitionSearch[T]) Search(rng *rand.Rand, size int, e *EntropyCounter[T]) []T {
	perm := make([]T, size)

	setGreedyPermutation := func(bit T) {
		diffSum := 0.0
		oldDiffSum := 0.0
		for i := range perm {
			pattern := T(i)
			if pattern&bit == 0 {
				other := pattern ^ bit
				diff := e.JointCount(pattern) - e.JointCount(other)
				diffSum += math.Abs(diff)
				oldDiffSum += diff
				if diff > 0 {
					perm[i], perm[other] = other, pattern
				} else {
					perm[i], perm[other] = pattern, other
				}
			}
		}
		if diffSum-math.Abs(oldDiffSum) < g.MinDifference {
			// Revert to identity for small sample-sized improvements
			for i := range perm {
				perm[i] = T(i)
			}
		}
	}

	bestObj := math.Inf(-1)
	var bestMask T
	for bitMask := uint64(1); (1 << bitMask) < size; bitMask <<= 1 {
		setGreedyPermutation(T(bitMask))
		obj := -e.PermutedBitwiseEntropy(perm)
		if obj > bestObj {
			bestObj = obj
			bestMask = T(bitMask)
		}
	}
	setGreedyPermutation(bestMask)
	return perm
}
