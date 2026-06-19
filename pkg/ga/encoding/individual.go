package encoding

import "github.com/tsp-solver/pkg/utils"

type EncodingType string

const (
	BinaryEncoding   EncodingType = "binary"
	RealEncoding     EncodingType = "real"
	PermutationEncoding EncodingType = "permutation"
	CustomEncoding   EncodingType = "custom"
)

type Individual struct {
	Genome    []float64
	Fitness   float64
	Evaluated bool
}

func NewIndividual(length int, encType EncodingType, permSize int, bounds [][2]float64) *Individual {
	ind := &Individual{
		Genome: make([]float64, length),
	}

	switch encType {
	case BinaryEncoding:
		for i := 0; i < length; i++ {
			if utils.RandFloat() < 0.5 {
				ind.Genome[i] = 0
			} else {
				ind.Genome[i] = 1
			}
		}
	case RealEncoding:
		for i := 0; i < length; i++ {
			if i < len(bounds) {
				low, high := bounds[i][0], bounds[i][1]
				ind.Genome[i] = low + utils.RandFloat()*(high-low)
			} else {
				ind.Genome[i] = utils.RandFloat()
			}
		}
	case PermutationEncoding:
		perm := utils.RandPerm(permSize)
		for i := 0; i < len(perm) && i < length; i++ {
			ind.Genome[i] = float64(perm[i])
		}
	}

	return ind
}

func (ind *Individual) Copy() *Individual {
	newInd := &Individual{
		Genome:    make([]float64, len(ind.Genome)),
		Fitness:   ind.Fitness,
		Evaluated: ind.Evaluated,
	}
	copy(newInd.Genome, ind.Genome)
	return newInd
}

func (ind *Individual) GetPermutation() []int {
	perm := make([]int, len(ind.Genome))
	for i, v := range ind.Genome {
		perm[i] = int(v)
	}
	return perm
}

func (ind *Individual) SetPermutation(perm []int) {
	for i, v := range perm {
		if i < len(ind.Genome) {
			ind.Genome[i] = float64(v)
		}
	}
}

type Population []*Individual

func NewPopulation(size, genomeLen int, encType EncodingType, permSize int, bounds [][2]float64) Population {
	pop := make(Population, size)
	for i := 0; i < size; i++ {
		pop[i] = NewIndividual(genomeLen, encType, permSize, bounds)
	}
	return pop
}

func (pop Population) Len() int {
	return len(pop)
}

func (pop Population) Less(i, j int) bool {
	return pop[i].Fitness > pop[j].Fitness
}

func (pop Population) Swap(i, j int) {
	pop[i], pop[j] = pop[j], pop[i]
}

func (pop Population) Best() *Individual {
	if len(pop) == 0 {
		return nil
	}
	best := pop[0]
	for _, ind := range pop[1:] {
		if ind.Fitness > best.Fitness {
			best = ind
		}
	}
	return best
}

func (pop Population) Worst() *Individual {
	if len(pop) == 0 {
		return nil
	}
	worst := pop[0]
	for _, ind := range pop[1:] {
		if ind.Fitness < worst.Fitness {
			worst = ind
		}
	}
	return worst
}

func (pop Population) AvgFitness() float64 {
	if len(pop) == 0 {
		return 0
	}
	sum := 0.0
	for _, ind := range pop {
		sum += ind.Fitness
	}
	return sum / float64(len(pop))
}

func (pop Population) FitnessVariance() float64 {
	if len(pop) <= 1 {
		return 0
	}
	avg := pop.AvgFitness()
	sum := 0.0
	for _, ind := range pop {
		diff := ind.Fitness - avg
		sum += diff * diff
	}
	return sum / float64(len(pop))
}

func (pop Population) Copy() Population {
	newPop := make(Population, len(pop))
	for i, ind := range pop {
		newPop[i] = ind.Copy()
	}
	return newPop
}
