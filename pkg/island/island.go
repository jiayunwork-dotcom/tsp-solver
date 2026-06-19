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
	ID                 int
	GA                 *ga.GeneticAlgorithm
	Config             ga.GAConfig
	ReceivedMigrations int
	ImprovedAfter      int
}

type IslandModel struct {
	Islands         []*Island
	MigrationPolicy MigrationPolicy
	NumIslands      int
	Gen             int
	TotalGens       int
	History         *ga.GAGenerationHistory
	Verbose         bool
}

func NewIslandModel(numIslands int, baseConfig ga.GAConfig, migrationPolicy MigrationPolicy, configVariants []ga.GAConfig, verbose bool) *IslandModel {
	islands := make([]*Island, numIslands)

	for i := 0; i < numIslands; i++ {
		config := baseConfig
		if i < len(configVariants) {
			config = configVariants[i]
		}

		islandGA := ga.NewGeneticAlgorithm(config)
		islands[i] = &Island{
			ID:                 i,
			GA:                 islandGA,
			Config:             config,
			ReceivedMigrations: 0,
			ImprovedAfter:      0,
		}
	}

	model := &IslandModel{
		Islands:         islands,
		MigrationPolicy: migrationPolicy,
		NumIslands:      numIslands,
		TotalGens:       baseConfig.Generations,
		History:         ga.NewGAGenerationHistory(),
		Verbose:         verbose,
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

	migrants := make([][]*encoding.Individual, model.NumIslands)
	prevBest := make([]float64, model.NumIslands)
	for i, island := range model.Islands {
		migrants[i] = model.selectMigrants(island, numMigrants)
		prevBest[i] = island.GA.BestIndividual().Fitness
	}

	for i, island := range model.Islands {
		neighborIdx := (i + 1) % model.NumIslands
		model.replaceWorst(island, migrants[neighborIdx])
		island.ReceivedMigrations++

		newBest := island.GA.BestIndividual().Fitness
		if newBest > prevBest[i]+1e-15 {
			island.ImprovedAfter++
		}
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

func (model *IslandModel) BestIslandID() int {
	bestID := 0
	var best *encoding.Individual
	for idx, island := range model.Islands {
		ib := island.GA.BestIndividual()
		if best == nil || ib.Fitness > best.Fitness {
			best = ib
			bestID = idx
		}
	}
	return bestID
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

	initialDiversity := 0.0
	if len(model.Islands) > 0 && len(model.Islands[0].GA.History.Diversity) > 0 {
		initialDiversity = model.Islands[0].GA.History.Diversity[0]
	}

	overallBestFitness := model.BestIndividual().Fitness
	firstBestGen := 0
	bestIslandID := model.BestIslandID()

	for model.Step() {
		curBest := model.BestIndividual().Fitness
		if curBest > overallBestFitness+1e-15 {
			overallBestFitness = curBest
			firstBestGen = model.Gen
			bestIslandID = model.BestIslandID()
		}

		if model.Verbose && model.Gen%10 == 0 {
			best := model.BestIndividual()
			println("Generation", model.Gen, "Best Fitness:", best.Fitness, "Best Island:", bestIslandID)
		}

		if model.Verbose && model.MigrationPolicy.Interval > 0 && model.Gen%model.MigrationPolicy.Interval == 0 {
			println("  Migration event", model.MigrationPolicy.MigrationCount, "completed")
		}
	}

	best := model.BestIndividual()
	bestIslandID = model.BestIslandID()

	finalDiversity := 0.0
	if len(model.History.Diversity) > 0 {
		finalDiversity = model.History.Diversity[len(model.History.Diversity)-1]
	}

	migrationStats := make([]ga.IslandMigrationStat, len(model.Islands))
	for i, island := range model.Islands {
		migrationStats[i] = ga.IslandMigrationStat{
			IslandID:           island.ID,
			ReceivedMigrations: island.ReceivedMigrations,
			ImprovedAfter:      island.ImprovedAfter,
		}
	}

	return &ga.GAResult{
		BestIndividual:      best,
		BestFitness:         best.Fitness,
		BestGeneration:      model.Gen,
		FirstBestGeneration: firstBestGen,
		History:             model.History,
		Duration:            time.Since(start),
		InitialDiversity:    initialDiversity,
		FinalDiversity:      finalDiversity,
		BestIslandID:        bestIslandID,
		IslandMigrationStats: migrationStats,
	}
}
