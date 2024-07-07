package groupcompress

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
