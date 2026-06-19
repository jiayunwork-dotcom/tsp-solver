package ga

import (
	"fmt"
	"time"

	"github.com/tsp-solver/pkg/ga/adaptive"
	"github.com/tsp-solver/pkg/ga/crossover"
	"github.com/tsp-solver/pkg/ga/encoding"
	"github.com/tsp-solver/pkg/ga/mutation"
	"github.com/tsp-solver/pkg/ga/selection"
	"github.com/tsp-solver/pkg/utils"
)

type FitnessFunc func(ind *encoding.Individual) float64

type GAConfig struct {
	EncodingType    encoding.EncodingType
	GenomeLength    int
	PermutationSize int
	Bounds          [][2]float64
	PopulationSize  int
	Generations     int
	EliteCount      int

	SelectionConfig selection.Config
	CrossoverConfig crossover.Config
	MutationConfig  mutation.Config
	AdaptiveConfig  adaptive.AdaptiveConfig

	FitnessFunction FitnessFunc
}

type GAResult struct {
	BestIndividual       *encoding.Individual
	BestFitness          float64
	BestGeneration       int
	FirstBestGeneration  int
	History              *GAGenerationHistory
	Duration             time.Duration
	LocalSearchCalls     int
	LocalSearchImproved  float64
	InitialDiversity     float64
	FinalDiversity       float64
	BestIslandID         int
	IslandMigrationStats []IslandMigrationStat
}

type IslandMigrationStat struct {
	IslandID           int
	ReceivedMigrations int
	ImprovedAfter      int
}

type GAGenerationHistory struct {
	Generations      []int
	BestFitness      []float64
	AvgFitness       []float64
	Diversity        []float64
	ImprovementRate  []float64
	StagnationCount  []int
}

func NewGAGenerationHistory() *GAGenerationHistory {
	return &GAGenerationHistory{
		Generations:     make([]int, 0),
		BestFitness:     make([]float64, 0),
		AvgFitness:      make([]float64, 0),
		Diversity:       make([]float64, 0),
		ImprovementRate: make([]float64, 0),
		StagnationCount: make([]int, 0),
	}
}

func (h *GAGenerationHistory) Add(gen int, best, avg, diversity float64) {
	h.Generations = append(h.Generations, gen)
	h.BestFitness = append(h.BestFitness, best)
	h.AvgFitness = append(h.AvgFitness, avg)
	h.Diversity = append(h.Diversity, diversity)

	idx := len(h.BestFitness) - 1
	if idx == 0 {
		h.ImprovementRate = append(h.ImprovementRate, 0.0)
		h.StagnationCount = append(h.StagnationCount, 0)
	} else {
		prevBest := h.BestFitness[idx-1]
		improveRate := 0.0
		if prevBest > 1e-15 {
			improveRate = (best - prevBest) / prevBest * 100.0
		}
		h.ImprovementRate = append(h.ImprovementRate, improveRate)

		if best > prevBest+1e-15 {
			h.StagnationCount = append(h.StagnationCount, 0)
		} else {
			h.StagnationCount = append(h.StagnationCount, h.StagnationCount[idx-1]+1)
		}
	}
}

type GeneticAlgorithm struct {
	Config  GAConfig
	Pop     encoding.Population
	Gen     int
	History *GAGenerationHistory
}

func NewGeneticAlgorithm(config GAConfig) *GeneticAlgorithm {
	ga := &GeneticAlgorithm{
		Config:  config,
		History: NewGAGenerationHistory(),
	}
	ga.initPopulation()
	return ga
}

func (ga *GeneticAlgorithm) initPopulation() {
	ga.Pop = encoding.NewPopulation(
		ga.Config.PopulationSize,
		ga.Config.GenomeLength,
		ga.Config.EncodingType,
		ga.Config.PermutationSize,
		ga.Config.Bounds,
	)
	ga.evaluatePopulation()

	best := ga.Pop.Best()
	avg := ga.Pop.AvgFitness()
	diversity := ga.calculateDiversity()
	ga.History.Add(0, best.Fitness, avg, diversity)
}

func (ga *GeneticAlgorithm) evaluatePopulation() {
	for _, ind := range ga.Pop {
		if !ind.Evaluated && ga.Config.FitnessFunction != nil {
			ind.Fitness = ga.Config.FitnessFunction(ind)
			ind.Evaluated = true
		}
	}
}

func (ga *GeneticAlgorithm) Step() bool {
	if ga.Gen >= ga.Config.Generations {
		return false
	}

	crossoverRate := ga.Config.CrossoverConfig.Rate
	mutationRate := ga.Config.MutationConfig.Rate

	if ga.Config.AdaptiveConfig.Enabled {
		rates := adaptive.CalculateAdaptiveRates(ga.Pop, ga.Config.AdaptiveConfig)
		crossoverRate = rates.CrossoverRate
		mutationRate = rates.MutationRate
	}

	elites := selection.ElitismSelect(ga.Pop, ga.Config.EliteCount)

	parents := selection.Select(ga.Pop, ga.Config.SelectionConfig)

	offspring := make(encoding.Population, 0, ga.Config.PopulationSize)

	for i := 0; i < len(parents); i += 2 {
		if i+1 >= len(parents) {
			offspring = append(offspring, parents[i].Copy())
			break
		}

		crossoverConfig := ga.Config.CrossoverConfig
		crossoverConfig.Rate = crossoverRate

		child1, child2 := crossover.Crossover(parents[i], parents[i+1], ga.Config.EncodingType, crossoverConfig)
		offspring = append(offspring, child1, child2)
	}

	for len(offspring) < ga.Config.PopulationSize {
		offspring = append(offspring, parents[utils.RandInt(0, len(parents)-1)].Copy())
	}
	offspring = offspring[:ga.Config.PopulationSize]

	mutationConfig := ga.Config.MutationConfig
	mutationConfig.Rate = mutationRate

	for i := range offspring {
		offspring[i] = mutation.Mutate(offspring[i], ga.Config.EncodingType, mutationConfig)
	}

	ga.evaluatePopulationOn(offspring)

	for i, elite := range elites {
		if i < len(offspring) {
			offspring[i] = elite.Copy()
		}
	}

	ga.Pop = offspring
	ga.Gen++

	best := ga.Pop.Best()
	avg := ga.Pop.AvgFitness()
	diversity := ga.calculateDiversity()

	ga.History.Add(ga.Gen, best.Fitness, avg, diversity)

	return true
}

func (ga *GeneticAlgorithm) evaluatePopulationOn(pop encoding.Population) {
	for _, ind := range pop {
		if !ind.Evaluated && ga.Config.FitnessFunction != nil {
			ind.Fitness = ga.Config.FitnessFunction(ind)
			ind.Evaluated = true
		}
	}
}

func (ga *GeneticAlgorithm) calculateDiversity() float64 {
	if len(ga.Pop) <= 1 {
		return 0
	}

	n := len(ga.Pop)
	totalDistance := 0.0
	count := 0

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			dist := ga.hammingDistance(ga.Pop[i], ga.Pop[j])
			totalDistance += dist
			count++
		}
	}

	if count == 0 {
		return 0
	}
	return totalDistance / float64(count)
}

func (ga *GeneticAlgorithm) hammingDistance(ind1, ind2 *encoding.Individual) float64 {
	if len(ind1.Genome) == 0 || len(ind2.Genome) == 0 {
		return 0
	}

	distance := 0.0
	minLen := len(ind1.Genome)
	if len(ind2.Genome) < minLen {
		minLen = len(ind2.Genome)
	}

	switch ga.Config.EncodingType {
	case encoding.BinaryEncoding:
		for i := 0; i < minLen; i++ {
			if ind1.Genome[i] != ind2.Genome[i] {
				distance += 1
			}
		}
	case encoding.PermutationEncoding:
		distance = float64(ga.editDistancePerm(ind1.GetPermutation(), ind2.GetPermutation()))
	case encoding.RealEncoding:
		for i := 0; i < minLen; i++ {
			distance += (ind1.Genome[i] - ind2.Genome[i]) * (ind1.Genome[i] - ind2.Genome[i])
		}
		distance /= float64(minLen)
	}

	return distance / float64(minLen)
}

func (ga *GeneticAlgorithm) editDistancePerm(p1, p2 []int) int {
	n := len(p1)
	if n != len(p2) {
		return -1
	}

	posMap := make(map[int]int)
	for i, v := range p2 {
		posMap[v] = i
	}

	distance := 0
	for i, v := range p1 {
		if posMap[v] != i {
			distance++
		}
	}

	return distance
}

func (ga *GeneticAlgorithm) Run() *GAResult {
	start := time.Now()

	for ga.Step() {
		if ga.Gen%10 == 0 {
			best := ga.Pop.Best()
			fmt.Printf("Generation %d, Best Fitness: %.6f, Avg: %.6f\n", ga.Gen, best.Fitness, ga.Pop.AvgFitness())
		}
	}

	best := ga.Pop.Best()
	bestGen := 0
	for i, fit := range ga.History.BestFitness {
		if fit >= best.Fitness && fit == best.Fitness {
			bestGen = i
			break
		}
	}

	return &GAResult{
		BestIndividual: best,
		BestFitness:    best.Fitness,
		BestGeneration: bestGen,
		History:        ga.History,
		Duration:       time.Since(start),
	}
}

func (ga *GeneticAlgorithm) BestIndividual() *encoding.Individual {
	return ga.Pop.Best()
}
