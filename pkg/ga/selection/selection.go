package selection

import (
	"sort"

	"github.com/tsp-solver/pkg/ga/encoding"
	"github.com/tsp-solver/pkg/utils"
)

type SelectionType string

const (
	RouletteWheel SelectionType = "roulette"
	Tournament    SelectionType = "tournament"
	Rank          SelectionType = "rank"
	Elitism       SelectionType = "elitism"
)

type Config struct {
	Type          SelectionType
	TournamentSize int
	EliteCount    int
}

func Select(pop encoding.Population, config Config) encoding.Population {
	switch config.Type {
	case RouletteWheel:
		return rouletteWheelSelect(pop, len(pop))
	case Tournament:
		return tournamentSelect(pop, len(pop), config.TournamentSize)
	case Rank:
		return rankSelect(pop, len(pop))
	default:
		return tournamentSelect(pop, len(pop), config.TournamentSize)
	}
}

func ElitismSelect(pop encoding.Population, eliteCount int) encoding.Population {
	sortedPop := pop.Copy()
	sort.Sort(sortedPop)
	elites := make(encoding.Population, eliteCount)
	for i := 0; i < eliteCount && i < len(sortedPop); i++ {
		elites[i] = sortedPop[i].Copy()
	}
	return elites
}

func rouletteWheelSelect(pop encoding.Population, n int) encoding.Population {
	offspring := make(encoding.Population, n)
	totalFitness := 0.0
	minFitness := pop[0].Fitness
	for _, ind := range pop {
		if ind.Fitness < minFitness {
			minFitness = ind.Fitness
		}
	}

	offset := 0.0
	if minFitness <= 0 {
		offset = -minFitness + 1.0
	}

	for _, ind := range pop {
		totalFitness += ind.Fitness + offset
	}

	for i := 0; i < n; i++ {
		r := utils.RandFloat() * totalFitness
		sum := 0.0
		for _, ind := range pop {
			sum += ind.Fitness + offset
			if sum >= r {
				offspring[i] = ind.Copy()
				break
			}
		}
		if offspring[i] == nil {
			offspring[i] = pop[len(pop)-1].Copy()
		}
	}
	return offspring
}

func tournamentSelect(pop encoding.Population, n, k int) encoding.Population {
	if k <= 0 {
		k = 3
	}
	offspring := make(encoding.Population, n)
	for i := 0; i < n; i++ {
		bestIdx := utils.RandInt(0, len(pop)-1)
		for j := 1; j < k; j++ {
			idx := utils.RandInt(0, len(pop)-1)
			if pop[idx].Fitness > pop[bestIdx].Fitness {
				bestIdx = idx
			}
		}
		offspring[i] = pop[bestIdx].Copy()
	}
	return offspring
}

func rankSelect(pop encoding.Population, n int) encoding.Population {
	sortedPop := pop.Copy()
	sort.Sort(sortedPop)

	totalRank := 0.0
	for i := range sortedPop {
		totalRank += float64(len(sortedPop) - i)
	}

	offspring := make(encoding.Population, n)
	for i := 0; i < n; i++ {
		r := utils.RandFloat() * totalRank
		sum := 0.0
		for j, ind := range sortedPop {
			sum += float64(len(sortedPop) - j)
			if sum >= r {
				offspring[i] = ind.Copy()
				break
			}
		}
		if offspring[i] == nil {
			offspring[i] = sortedPop[len(sortedPop)-1].Copy()
		}
	}
	return offspring
}
