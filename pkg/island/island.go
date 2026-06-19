package island

import (
	"sort"
	"time"

	"github.com/tsp-solver/pkg/ga"
	"github.com/tsp-solver/pkg/ga/encoding"
	"github.com/tsp-solver/pkg/utils"
)

type MigrationPolicy struct {
	Interval       int
	NumMigrants    int
	Topology       string
	MigrationCount int
}

type Island struct {
	ID     int
	GA     *ga.GeneticAlgorithm
	Config ga.GAConfig
}

type IslandModel struct {
	Islands         []*Island
	MigrationPolicy MigrationPolicy
	NumIslands      int
	Gen             int
	TotalGens       int
	History         *ga.GAGenerationHistory
}

func NewIslandModel(numIslands int, baseConfig ga.GAConfig, migrationPolicy MigrationPolicy, configVariants []ga.GAConfig) *IslandModel {
	islands := make([]*Island, numIslands)

	for i := 0; i < numIslands; i++ {
		config := baseConfig
		if i < len(configVariants) {
			config = configVariants[i]
		}

		islandGA := ga.NewGeneticAlgorithm(config)
		islands[i] = &Island{
			ID:     i,
			GA:     islandGA,
			Config: config,
		}
	}

	model := &IslandModel{
		Islands:         islands,
		MigrationPolicy: migrationPolicy,
		NumIslands:      numIslands,
		TotalGens:       baseConfig.Generations,
		History:         ga.NewGAGenerationHistory(),
	}

	return model
}

func (model *IslandModel) Step() bool {
	if model.Gen >= model.TotalGens {
		return false
	}

	for _, island := range model.Islands {
		island.GA.Step()
	}

	model.Gen++

	if model.MigrationPolicy.Interval > 0 && model.Gen%model.MigrationPolicy.Interval == 0 {
		model.migrate()
		model.MigrationPolicy.MigrationCount++
	}

	bestOverall := model.BestIndividual()
	avgFitness := model.AvgFitness()
	diversity := model.OverallDiversity()

	model.History.Add(model.Gen, bestOverall.Fitness, avgFitness, diversity)

	return true
}

func (model *IslandModel) migrate() {
	switch model.MigrationPolicy.Topology {
	case "ring":
		model.migrateRing()
	default:
		model.migrateRing()
	}
}

func (model *IslandModel) migrateRing() {
	numMigrants := model.MigrationPolicy.NumMigrants
	if numMigrants <= 0 {
		numMigrants = 1
	}

	migrants := make([][] *encoding.Individual, model.NumIslands)
	for i, island := range model.Islands {
		migrants[i] = model.selectMigrants(island, numMigrants)
	}

	for i, island := range model.Islands {
		neighborIdx := (i + 1) % model.NumIslands
		model.replaceWorst(island, migrants[neighborIdx])
	}
}

func (model *IslandModel) selectMigrants(island *Island, numMigrants int) []*encoding.Individual {
	pop := island.GA.Pop
	sortedPop := pop.Copy()
	sort.Sort(sortedPop)

	selected := make([]*encoding.Individual, numMigrants)
	topCount := len(sortedPop) / 2
	if topCount < numMigrants {
		topCount = numMigrants
	}
	if topCount > len(sortedPop) {
		topCount = len(sortedPop)
	}

	indices := utils.RandPerm(topCount)
	for i := 0; i < numMigrants && i < len(indices); i++ {
		selected[i] = sortedPop[indices[i]].Copy()
	}

	return selected
}

func (model *IslandModel) replaceWorst(island *Island, migrants []*encoding.Individual) {
	pop := island.GA.Pop
	sortedPop := make(encoding.Population, len(pop))
	copy(sortedPop, pop)
	sort.Sort(sortedPop)

	worstIndices := make([]int, 0, len(migrants))
	for i := len(sortedPop) - 1; i >= 0 && len(worstIndices) < len(migrants); i-- {
		for j, ind := range pop {
			if ind == sortedPop[i] {
				already := false
				for _, idx := range worstIndices {
					if idx == j {
						already = true
						break
					}
				}
				if !already {
					worstIndices = append(worstIndices, j)
					break
				}
			}
		}
	}

	for i, migrant := range migrants {
		if i < len(worstIndices) {
			pop[worstIndices[i]] = migrant.Copy()
		}
	}
}

func (model *IslandModel) BestIndividual() *encoding.Individual {
	var best *encoding.Individual
	for _, island := range model.Islands {
		ib := island.GA.BestIndividual()
		if best == nil || ib.Fitness > best.Fitness {
			best = ib
		}
	}
	return best
}

func (model *IslandModel) AvgFitness() float64 {
	total := 0.0
	count := 0
	for _, island := range model.Islands {
		total += island.GA.Pop.AvgFitness()
		count++
	}
	if count == 0 {
		return 0
	}
	return total / float64(count)
}

func (model *IslandModel) OverallDiversity() float64 {
	totalDiversity := 0.0
	for _, island := range model.Islands {
		div := 0.0
		if len(island.GA.History.Diversity) > 0 {
			div = island.GA.History.Diversity[len(island.GA.History.Diversity)-1]
		}
		totalDiversity += div
	}
	return totalDiversity / float64(len(model.Islands))
}

func (model *IslandModel) Run() *ga.GAResult {
	start := time.Now()

	for model.Step() {
		if model.Gen%10 == 0 {
			best := model.BestIndividual()
			println("Generation", model.Gen, "Best Fitness:", best.Fitness)
		}
	}

	best := model.BestIndividual()
	bestGen := 0
	for i, fit := range model.History.BestFitness {
		if fit >= best.Fitness {
			bestGen = i
			break
		}
	}

	return &ga.GAResult{
		BestIndividual: best,
		BestFitness:    best.Fitness,
		BestGeneration: bestGen,
		History:        model.History,
		Duration:       time.Since(start),
	}
}
