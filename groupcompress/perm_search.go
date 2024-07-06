package groupcompress

import (
	"math/rand"
	"sort"
)

// A PermObjectiveFunc returns a deterministic value for a
// permutation over N elements.
//
// Objective functions are not necessarily safe to call
// concurrently from multiple Goroutines.
type PermObjectiveFunc[T BitPattern] func(p []T) float64

// A PermSearch implements an algorithm for optimizing an
// objective over the space of permutations.
type PermSearch[T BitPattern] interface {
	// Search finds a permutation to maximize f.
	// The size argument indicates the number of elements
	// in the permutation.
	// The rng may be used as part of the search.
	Search(rng *rand.Rand, size int, f PermObjectiveFunc[T]) []T
}

// RandomPermSearch is a PermSearch which samples random
// permutations and selects the best one.
type RandomPermSearch[T BitPattern] struct {
	Samples int
}

// Search tries r.Samples permutations and returns the one
// with the best objective value.
func (r *RandomPermSearch[T]) Search(rng *rand.Rand, size int, f PermObjectiveFunc[T]) []T {
	var bestPerm []T
	var bestObj float64
	for i := 0; i < r.Samples; i++ {
		perm := make([]T, size)
		for i, x := range rng.Perm(len(perm)) {
			perm[i] = T(x)
		}
		obj := f(perm)
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
func (e *EvoPermSearch[T]) Search(rng *rand.Rand, size int, f PermObjectiveFunc[T]) []T {
	if e.Generations == 0 {
		return (&RandomPermSearch[T]{
			Samples: e.Population * (1 + e.Mutations),
		}).Search(rng, size, f)
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
		p.Value = f(p.Perm)
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
				child.Value = f(child.Perm)
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
