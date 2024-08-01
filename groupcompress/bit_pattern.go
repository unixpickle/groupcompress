package groupcompress

type BitPattern uint64

func ExtractBitPattern(s *BitString, indices []int) BitPattern {
	var result BitPattern
	for i, j := range indices {
		bit := s.Get(j)
		if bit {
			result |= 1 << i
		}
	}
	return result
}
