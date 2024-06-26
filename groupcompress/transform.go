package groupcompress

// A Transform gathers a subset of bits from a string, uses
// a lookup table to transform them to a different subset,
// and then writes the result back to the original location.
//
// The type T should be large enough to hold len(Indices)
// bits, and typically a uint8 should be sufficient.
type Transform[T BitPattern] struct {
	// Indices indices the bits to lookup in an input string.
	Indices []int

	// Mapping maps each source bit pattern to a target bit
	// pattern. Should be of size 2^len(Indices).
	Mapping []T
}

// Inverse gets the inverse transform.
func (t *Transform[T]) Inverse() *Transform[T] {
	return &Transform[T]{
		Indices: t.Indices,
		Mapping: InvertMapping[T](t.Mapping),
	}
}

// Apply transforms the bit string using the lookup table.
func (t *Transform[T]) Apply(bs *BitString) {
	var value T
	for i, idx := range t.Indices {
		if bs.Get(idx) {
			value |= 1 << uint(i)
		}
	}
	newValue := t.Mapping[value]
	for i, idx := range t.Indices {
		bs.Set(idx, newValue&(1<<uint(i)) != 0)
	}
}

func InvertMapping[T BitPattern](mapping []T) []T {
	reverse := make([]T, len(mapping))
	for i, x := range mapping {
		reverse[x] = T(i)
	}
	return reverse
}
