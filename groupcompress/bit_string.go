package groupcompress

// A BitString is an array of bits.
type BitString struct {
	NumBits int
	Bitmap  []byte
}

// NewBitString creates a bit string of all 0's.
func NewBitString(numBits int) *BitString {
	numBytes := numBits / 8
	if numBits%8 != 0 {
		numBytes++
	}
	return &BitString{
		NumBits: numBits,
		Bitmap:  make([]byte, numBytes),
	}
}

// Get returns the bit at the given index.
func (d *BitString) Get(index int) bool {
	if index < 0 || index >= d.NumBits {
		panic("index out of range")
	}
	byteIndex := index >> 3
	bitIndex := uint(index & 7)
	return d.Bitmap[byteIndex]&(1<<bitIndex) != 0
}

// Set sets the bit at the given index.
func (d *BitString) Set(index int, value bool) {
	if index < 0 || index >= d.NumBits {
		panic("index out of range")
	}
	byteIndex := index / 8
	bitIndex := uint(index % 8)
	if value {
		d.Bitmap[byteIndex] |= (1 << bitIndex)
	} else {
		d.Bitmap[byteIndex] &= ^byte((1 << bitIndex))
	}
}
