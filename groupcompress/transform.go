package groupcompress

// A Transform gathers a subset of bits from a string, uses
// a lookup table to transform them to a different subset,
// and then writes the result back to the original location.
type Transform struct {
	// Indices indices the bits to lookup in an input string.
	Indices []int

	// Mapping maps each source bit pattern to a target bit
	// pattern. Should be of size 2^len(Indices).
	Mapping Perm
}

// Inverse gets the inverse transform.
func (t *Transform) Inverse() *Transform {
	return &Transform{
		Indices: t.Indices,
		Mapping: t.Mapping.Inverse(),
	}
}

// Apply transforms the bit string using the lookup table.
func (t *Transform) Apply(bs *BitString) {
	var value BitPattern
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

type TransformList []*Transform

func (t TransformList) Inverse() TransformList {
	result := make(TransformList, 0, len(t))
	for i := len(t) - 1; i >= 0; i-- {
		result = append(result, t[i].Inverse())
	}
	return result
}

func (t TransformList) Apply(b *BitString) {
	for _, x := range t {
		x.Apply(b)
	}
}
