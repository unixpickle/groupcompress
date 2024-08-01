package groupcompress

import "encoding/json"

// A Transform gathers a subset of bits from a string, uses
// a lookup table to transform them to a different subset,
// and then writes the result back to the original location.
type Transform struct {
	// Indices indices the bits to lookup in an input string.
	Indices []int

	// Mapping maps each source bit pattern to a target bit
	// pattern. Should be of size 2^len(Indices).
	Mapping []BitPattern
}

func (t *Transform) MarshalJSON() ([]byte, error) {
	if len(t.Mapping) > 256 {
		return json.Marshal(&struct {
			Indices []int
			Mapping []BitPattern
		}{t.Indices, t.Mapping})
	} else {
		mapping8 := make([]uint8, len(t.Mapping))
		for i, x := range t.Mapping {
			mapping8[i] = uint8(x)
		}
		return json.Marshal(&struct {
			Indices []int
			Mapping []uint8
		}{t.Indices, mapping8})
	}
}

func (t *Transform) UnmarshalJSON(data []byte) error {
	var obj struct {
		Indices []int
		Mapping []uint8
	}
	if json.Unmarshal(data, &obj) == nil {
		t.Indices = obj.Indices
		t.Mapping = make([]BitPattern, len(obj.Mapping))
		for i, x := range obj.Mapping {
			t.Mapping[i] = BitPattern(x)
		}
		return nil
	}

	var obj1 struct {
		Indices []int
		Mapping []BitPattern
	}
	if err := json.Unmarshal(data, &obj1); err != nil {
		return err
	}
	t.Indices = obj1.Indices
	t.Mapping = obj1.Mapping
	return nil
}

// Inverse gets the inverse transform.
func (t *Transform) Inverse() *Transform {
	return &Transform{
		Indices: t.Indices,
		Mapping: InvertMapping(t.Mapping),
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

func InvertMapping(mapping []BitPattern) []BitPattern {
	reverse := make([]BitPattern, len(mapping))
	for i, x := range mapping {
		reverse[x] = BitPattern(i)
	}
	return reverse
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
