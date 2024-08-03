package groupcompress

import "encoding/json"

// A Perm represents a permutation over a space of bit
// patterns.
//
// The length of a Perm should always be 2**n for some
// integer n.
type Perm []BitPattern

func (p *Perm) MarshalJSON() ([]byte, error) {
	if len(*p) > 256 {
		return json.Marshal([]BitPattern(*p))
	} else {
		mapping8 := make([]uint8, len(*p))
		for i, x := range *p {
			mapping8[i] = uint8(x)
		}
		return json.Marshal(mapping8)
	}
}

func (p *Perm) UnmarshalJSON(data []byte) error {
	var obj []byte
	if json.Unmarshal(data, &obj) == nil {
		*p = make(Perm, len(obj))
		for i, x := range obj {
			(*p)[i] = BitPattern(x)
		}
		return nil
	}

	var obj1 []BitPattern
	if err := json.Unmarshal(data, &obj1); err != nil {
		return err
	}
	*p = obj1
	return nil
}

func (p *Perm) Inverse() Perm {
	reverse := make([]BitPattern, len(*p))
	for i, x := range *p {
		reverse[x] = BitPattern(i)
	}
	return reverse
}
