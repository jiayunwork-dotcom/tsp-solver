package mutation

import (
	"math"

	"github.com/tsp-solver/pkg/ga/encoding"
	"github.com/tsp-solver/pkg/utils"
)

type MutationType string

const (
	SwapMutation     MutationType = "swap"
	InversionMutation MutationType = "inversion"
	InsertMutation   MutationType = "insert"

	BitFlip MutationType = "bit_flip"

	Gaussian MutationType = "gaussian"
)

type Config struct {
	Type        MutationType
	Rate        float64
	GaussianStd float64
	Bounds      [][2]float64
}

func Mutate(ind *encoding.Individual, encType encoding.EncodingType, config Config) *encoding.Individual {
	mutated := ind.Copy()

	switch encType {
	case encoding.PermutationEncoding:
		permutationMutate(mutated, config)
	case encoding.BinaryEncoding:
		binaryMutate(mutated, config)
	case encoding.RealEncoding:
		realMutate(mutated, config)
	}

	mutated.Evaluated = false
	return mutated
}

func permutationMutate(ind *encoding.Individual, config Config) {
	perm := ind.GetPermutation()
	n := len(perm)

	switch config.Type {
	case SwapMutation:
		if utils.RandFloat() < config.Rate {
			i := utils.RandInt(0, n-1)
			j := utils.RandInt(0, n-1)
			perm[i], perm[j] = perm[j], perm[i]
		}
	case InversionMutation:
		if utils.RandFloat() < config.Rate {
			i := utils.RandInt(0, n-2)
			j := utils.RandInt(i+1, n-1)
			for k := 0; k < (j-i+1)/2; k++ {
				perm[i+k], perm[j-k] = perm[j-k], perm[i+k]
			}
		}
	case InsertMutation:
		if utils.RandFloat() < config.Rate {
			i := utils.RandInt(0, n-1)
			j := utils.RandInt(0, n-1)
			if i == j {
				return
			}
			val := perm[i]
			if i < j {
				for k := i; k < j; k++ {
					perm[k] = perm[k+1]
				}
			} else {
				for k := i; k > j; k-- {
					perm[k] = perm[k-1]
				}
			}
			perm[j] = val
		}
	default:
		if utils.RandFloat() < config.Rate {
			i := utils.RandInt(0, n-1)
			j := utils.RandInt(0, n-1)
			perm[i], perm[j] = perm[j], perm[i]
		}
	}

	ind.SetPermutation(perm)
}

func binaryMutate(ind *encoding.Individual, config Config) {
	for i := range ind.Genome {
		if utils.RandFloat() < config.Rate {
			if ind.Genome[i] == 0 {
				ind.Genome[i] = 1
			} else {
				ind.Genome[i] = 0
			}
		}
	}
}

func realMutate(ind *encoding.Individual, config Config) {
	std := config.GaussianStd
	if std <= 0 {
		std = 0.1
	}

	for i := range ind.Genome {
		if utils.RandFloat() < config.Rate {
			ind.Genome[i] += utils.Gaussian(0, std)

			if i < len(config.Bounds) {
				low := config.Bounds[i][0]
				high := config.Bounds[i][1]
				if ind.Genome[i] < low {
					ind.Genome[i] = low
				}
				if ind.Genome[i] > high {
					ind.Genome[i] = high
				}
			} else {
				ind.Genome[i] = math.Max(0, math.Min(1, ind.Genome[i]))
			}
		}
	}
}
