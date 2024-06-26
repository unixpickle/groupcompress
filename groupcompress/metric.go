package groupcompress

import "math"

func MeanBitwiseEntropy(data []*BitString) float64 {
	singleCounts := make([]int, data[0].NumBits)
	for _, datum := range data {
		for j := 0; j < datum.NumBits; j++ {
			bit := datum.Get(j)
			if bit {
				singleCounts[j]++
			}
		}
	}
	var bitwiseEntropy float64
	for _, count := range singleCounts {
		if count == 0 || count == len(data) {
			continue
		}
		probTrue := float64(count) / float64(len(data))
		probFalse := 1 - probTrue
		bitwiseEntropy -= probTrue*math.Log(probTrue) + probFalse*math.Log(probFalse)
	}
	return bitwiseEntropy / float64(len(singleCounts))
}
