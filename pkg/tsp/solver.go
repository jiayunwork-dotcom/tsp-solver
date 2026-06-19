package tsp

import (
	"sort"
	"time"

	"github.com/tsp-solver/pkg/config"
	"github.com/tsp-solver/pkg/ga"
	"github.com/tsp-solver/pkg/ga/adaptive"
	"github.com/tsp-solver/pkg/ga/crossover"
	"github.com/tsp-solver/pkg/ga/encoding"
	"github.com/tsp-solver/pkg/ga/mutation"
	"github.com/tsp-solver/pkg/ga/selection"
	"github.com/tsp-solver/pkg/local_search"
)

type TSPSolver struct {
	Problem             *TSPProblem
	Config              *config.GAConfig
	GA                  *ga.GeneticAlgorithm
	KDTree              *local_search.KDTree
	Points              []local_search.Point
	DistFunc            local_search.DistanceFunc
	LocalSearchCalls    int
	LocalSearchImproved float64
}

func NewTSPSolver(problem *TSPProblem, cfg *config.GAConfig) *TSPSolver {
	solver := &TSPSolver{
		Problem: problem,
		Config:  cfg,
	}

	solver.Points = make([]local_search.Point, problem.NumCities)
	for i, c := range problem.Cities {
		solver.Points[i] = local_search.Point{X: c.X, Y: c.Y, ID: i}
	}

	solver.DistFunc = func(i, j int) float64 {
		return problem.Distance(i, j)
	}

	if cfg.LocalSearch.Enabled && cfg.LocalSearch.UseKDTree && problem.NumCities >= cfg.LocalSearch.KDTreeThreshold {
		solver.KDTree = local_search.BuildKDTree(solver.Points)
	}

	return solver
}

func (s *TSPSolver) createGAConfig() ga.GAConfig {
	cfg := s.Config

	fitnessFunc := func(ind *encoding.Individual) float64 {
		tour := ind.GetPermutation()
		return s.Problem.Fitness(tour)
	}

	encType := encoding.EncodingType(cfg.Encoding)
	if encType == "" {
		encType = encoding.PermutationEncoding
	}

	permutationSize := s.Problem.NumCities
	if cfg.PermutationSize > 0 {
		permutationSize = cfg.PermutationSize
	}

	return ga.GAConfig{
		EncodingType:    encType,
		GenomeLength:    permutationSize,
		PermutationSize: permutationSize,
		Bounds:          nil,
		PopulationSize:  cfg.PopulationSize,
		Generations:     cfg.Generations,
		EliteCount:      cfg.Selection.EliteCount,

		SelectionConfig: selection.Config{
			Type:           selection.SelectionType(cfg.Selection.Type),
			TournamentSize: cfg.Selection.TournamentSize,
			EliteCount:     cfg.Selection.EliteCount,
		},
		CrossoverConfig: crossover.Config{
			Type:    crossover.CrossoverType(cfg.Crossover.Type),
			Rate:    cfg.Crossover.Rate,
			SBX_eta: cfg.Crossover.SBX_Eta,
		},
		MutationConfig: mutation.Config{
			Type:        mutation.MutationType(cfg.Mutation.Type),
			Rate:        cfg.Mutation.Rate,
			GaussianStd: cfg.Mutation.GaussianStd,
		},
		AdaptiveConfig: adaptive.AdaptiveConfig{
			Enabled:           cfg.Adaptive.Enabled,
			BaseCrossoverRate: cfg.Adaptive.BaseCrossoverRate,
			BaseMutationRate:  cfg.Adaptive.BaseMutationRate,
			MinCrossoverRate:  cfg.Adaptive.MinCrossoverRate,
			MaxCrossoverRate:  cfg.Adaptive.MaxCrossoverRate,
			MinMutationRate:   cfg.Adaptive.MinMutationRate,
			MaxMutationRate:   cfg.Adaptive.MaxMutationRate,
			VarianceThreshold: cfg.Adaptive.VarianceThreshold,
			Sensitivity:       cfg.Adaptive.Sensitivity,
		},

		FitnessFunction: fitnessFunc,
	}
}

func (s *TSPSolver) Solve() *ga.GAResult {
	start := time.Now()

	gaConfig := s.createGAConfig()
	s.GA = ga.NewGeneticAlgorithm(gaConfig)

	totalGens := s.Config.Generations
	localSearchInterval := s.Config.LocalSearch.Interval
	if localSearchInterval <= 0 {
		localSearchInterval = s.Config.Generations + 1
	}

	localSearchEnabled := s.Config.LocalSearch.Enabled

	initialDiversity := 0.0
	if len(s.GA.History.Diversity) > 0 {
		initialDiversity = s.GA.History.Diversity[0]
	}

	overallBestFitness := s.GA.BestIndividual().Fitness
	firstBestGen := 0

	for s.GA.Gen < totalGens {
		s.GA.Step()

		curBest := s.GA.BestIndividual().Fitness
		if curBest > overallBestFitness+1e-15 {
			overallBestFitness = curBest
			firstBestGen = s.GA.Gen
		}

		if localSearchEnabled && s.GA.Gen > 0 && s.GA.Gen%localSearchInterval == 0 {
			s.applyLocalSearch()
			curBest = s.GA.BestIndividual().Fitness
			if curBest > overallBestFitness+1e-15 {
				overallBestFitness = curBest
				firstBestGen = s.GA.Gen
			}
		}

		if s.Config.Output.Verbose && s.GA.Gen%10 == 0 {
			best := s.GA.BestIndividual()
			tourLen := 1.0 / best.Fitness
			println("Gen", s.GA.Gen, "Best distance:", tourLen)
		}
	}

	if localSearchEnabled {
		s.applyLocalSearch()
		curBest := s.GA.BestIndividual().Fitness
		if curBest > overallBestFitness+1e-15 {
			overallBestFitness = curBest
			firstBestGen = s.GA.Gen
		}
	}

	best := s.GA.BestIndividual()

	finalDiversity := 0.0
	if len(s.GA.History.Diversity) > 0 {
		finalDiversity = s.GA.History.Diversity[len(s.GA.History.Diversity)-1]
	}

	return &ga.GAResult{
		BestIndividual:      best,
		BestFitness:         best.Fitness,
		BestGeneration:      s.GA.Gen,
		FirstBestGeneration: firstBestGen,
		History:             s.GA.History,
		Duration:            time.Since(start),
		LocalSearchCalls:    s.LocalSearchCalls,
		LocalSearchImproved: s.LocalSearchImproved,
		InitialDiversity:    initialDiversity,
		FinalDiversity:      finalDiversity,
	}
}

func (s *TSPSolver) applyLocalSearch() {
	s.LocalSearchCalls++

	pop := s.GA.Pop
	sortedPop := pop.Copy()
	sort.Sort(sortedPop)

	topK := s.Config.LocalSearch.TopK
	if topK > len(sortedPop) {
		topK = len(sortedPop)
	}

	lsType := s.Config.LocalSearch.Type
	useKDTree := s.Config.LocalSearch.UseKDTree && s.Problem.NumCities >= s.Config.LocalSearch.KDTreeThreshold

	totalImproved := 0.0

	for i := 0; i < topK; i++ {
		oldLength := 1.0 / sortedPop[i].Fitness
		tour := sortedPop[i].GetPermutation()
		var newTour []int
		var newLength float64

		switch lsType {
		case "2-opt", "2opt":
			if useKDTree && s.KDTree != nil {
				newTour, newLength = local_search.TwoOptFast(tour, s.DistFunc, s.Points, s.KDTree, s.Config.LocalSearch.KDTreeNeighbors)
			} else {
				newTour, newLength = local_search.TwoOpt(tour, s.DistFunc)
			}
		case "3-opt", "3opt":
			newTour, newLength = local_search.ThreeOpt(tour, s.DistFunc)
		case "or-opt", "or_opt", "oropt":
			newTour, newLength = local_search.OrOpt(tour, s.DistFunc)
		default:
			if useKDTree && s.KDTree != nil {
				newTour, newLength = local_search.TwoOptFast(tour, s.DistFunc, s.Points, s.KDTree, s.Config.LocalSearch.KDTreeNeighbors)
			} else {
				newTour, newLength = local_search.TwoOpt(tour, s.DistFunc)
			}
		}

		if newLength < oldLength {
			totalImproved += oldLength - newLength
			sortedPop[i].SetPermutation(newTour)
			sortedPop[i].Fitness = 1.0 / newLength
			sortedPop[i].Evaluated = true
		}
	}

	s.LocalSearchImproved += totalImproved

	for i := 0; i < topK && i < len(pop); i++ {
		found := false
		for j := range pop {
			if pop[j].Fitness == sortedPop[i].Fitness {
				pop[j] = sortedPop[i]
				found = true
				break
			}
		}
		if !found {
			worstIdx := 0
			for j, ind := range pop {
				if ind.Fitness < pop[worstIdx].Fitness {
					worstIdx = j
				}
			}
			pop[worstIdx] = sortedPop[i]
		}
	}
}

func (s *TSPSolver) GetBestTour() []int {
	best := s.GA.BestIndividual()
	return best.GetPermutation()
}

func (s *TSPSolver) GetBestDistance() float64 {
	best := s.GA.BestIndividual()
	return 1.0 / best.Fitness
}
