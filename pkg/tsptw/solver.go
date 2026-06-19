package tsptw

import (
	"fmt"
	"sort"
	"time"

	"github.com/tsp-solver/pkg/config"
	"github.com/tsp-solver/pkg/utils"
)

type TSPTWSolver struct {
	Problem       *TSPTWProblem
	Config        *config.TSPTWConfig
	Penalty       *PenaltyController
	Population    [][]int
	Fitnesses     []float64
	Gen           int
	BestTour      []int
	BestCost      float64
	Convergence   []ConvergencePoint
}

type ConvergencePoint struct {
	Generation     int
	BestCost       float64
	FeasibleRatio  float64
}

type TSPTWResult struct {
	BestTour       []int
	BestCost       float64
	BestEval       *TourEvaluation
	Duration       time.Duration
	Generations    int
	FeasibleRatio  float64
	FinalPenalty   float64
	Convergence    []ConvergencePoint
}

func NewTSPTWSolver(problem *TSPTWProblem, cfg *config.TSPTWConfig) *TSPTWSolver {
	return &TSPTWSolver{
		Problem: problem,
		Config:  cfg,
		Penalty: NewPenaltyController(
			cfg.PenaltyType,
			cfg.PenaltyCoefficient,
			cfg.FeasibilityTarget,
			cfg.PenaltyAdjustInterval,
		),
	}
}

func (s *TSPTWSolver) Solve() *TSPTWResult {
	start := time.Now()

	s.initPopulation()

	totalGens := s.Config.Generations
	s.Convergence = make([]ConvergencePoint, 0, totalGens+1)

	s.recordConvergence()

	for s.Gen < totalGens {
		s.step()
		s.recordConvergence()

		if s.Config.PenaltyType == "adaptive" && s.Gen > 0 &&
			s.Gen%s.Penalty.PenaltyAdjustInterval == 0 {
			oldCoeff := s.Penalty.GetCoefficient()
			ratio := ComputeFeasibleRatio(s.Problem, s.Population)
			s.Penalty.Adjust(ratio)
			newCoeff := s.Penalty.GetCoefficient()
			if s.Config.Verbose {
				direction := "unchanged"
				if newCoeff > oldCoeff+1e-10 {
					direction = "increased"
				} else if newCoeff < oldCoeff-1e-10 {
					direction = "decreased"
				}
				fmt.Printf("Gen %d: Penalty adjustment [adaptive] | feasible_ratio=%.4f (target=%.4f) | coeff %.2f -> %.2f (%s)\n",
					s.Gen, ratio, s.Penalty.FeasibilityTarget, oldCoeff, newCoeff, direction)
			}
			s.recomputeAllFitness()
		}

		if s.Config.RepairEnabled && s.Gen > 0 &&
			s.Gen%s.Config.RepairInterval == 0 {
			popCopy := make([][]int, len(s.Population))
			copy(popCopy, s.Population)
			RepairTopK(s.Problem, popCopy, s.Config.RepairTopK)
			s.Population = popCopy
			s.recomputeAllFitness()
			if s.Config.Verbose {
				ratio := ComputeFeasibleRatio(s.Problem, s.Population)
				fmt.Printf("Gen %d: Repair applied (top-%d), feasible_ratio=%.4f\n",
					s.Gen, s.Config.RepairTopK, ratio)
			}
		}

		if s.Config.Verbose && s.Gen%50 == 0 {
			eval := s.Problem.EvaluateTour(s.BestTour)
			fmt.Printf("Gen %d: cost=%.2f dist=%.2f wait=%.2f violation=%.2f penalty_coeff=%.2f feasible=%.4f\n",
				s.Gen, s.BestCost, eval.TotalDistance, eval.TotalWaitTime, eval.TotalViolation,
				s.Penalty.GetCoefficient(), ComputeFeasibleRatio(s.Problem, s.Population))
		}
	}

	eval := s.Problem.EvaluateTour(s.BestTour)
	bestTourNormalized := ensureDepotFirst(s.BestTour)
	feasibleRatio := ComputeFeasibleRatio(s.Problem, s.Population)

	return &TSPTWResult{
		BestTour:      bestTourNormalized,
		BestCost:      s.BestCost,
		BestEval:      eval,
		Duration:      time.Since(start),
		Generations:   totalGens,
		FeasibleRatio: feasibleRatio,
		FinalPenalty:  s.Penalty.GetCoefficient(),
		Convergence:   s.Convergence,
	}
}

func (s *TSPTWSolver) initPopulation() {
	popSize := s.Config.PopulationSize
	n := s.Problem.NumCities

	s.Population = make([][]int, popSize)
	s.Fitnesses = make([]float64, popSize)

	for i := 0; i < popSize; i++ {
		s.Population[i] = randomTour(n)
		s.Fitnesses[i] = s.Problem.ComputePenalizedFitness(s.Population[i], s.Penalty.GetCoefficient())
	}

	s.updateBest()
	s.Gen = 0
}

func (s *TSPTWSolver) step() {
	popSize := s.Config.PopulationSize
	eliteCount := s.Config.EliteCount

	type idxFit struct {
		idx int
		fit float64
	}
	ranked := make([]idxFit, popSize)
	for i, f := range s.Fitnesses {
		ranked[i] = idxFit{idx: i, fit: f}
	}
	sort.Slice(ranked, func(i, j int) bool {
		return ranked[i].fit > ranked[j].fit
	})

	elites := make([][]int, 0, eliteCount)
	eliteFits := make([]float64, 0, eliteCount)
	for i := 0; i < eliteCount && i < len(ranked); i++ {
		idx := ranked[i].idx
		elites = append(elites, copyTour(s.Population[idx]))
		eliteFits = append(eliteFits, s.Fitnesses[idx])
	}

	offspring := make([][]int, 0, popSize)
	offspringFits := make([]float64, 0, popSize)

	for len(offspring) < popSize {
		p1Idx := tournamentSelect(s.Population, s.Fitnesses, s.Config.TournamentSize)
		p2Idx := tournamentSelect(s.Population, s.Fitnesses, s.Config.TournamentSize)

		var child1, child2 []int
		if utils.RandFloat() < s.Config.CrossoverRate {
			child1, child2 = oxCrossover(s.Population[p1Idx], s.Population[p2Idx])
		} else {
			child1 = copyTour(s.Population[p1Idx])
			child2 = copyTour(s.Population[p2Idx])
		}

		child1 = normalizeTour(child1, s.Problem.NumCities)
		child2 = normalizeTour(child2, s.Problem.NumCities)

		if utils.RandFloat() < s.Config.MutationRate {
			child1 = swapMutate(child1)
		}
		if utils.RandFloat() < s.Config.MutationRate {
			child2 = insertMutate(child2)
		}

		child1 = normalizeTour(child1, s.Problem.NumCities)
		child2 = normalizeTour(child2, s.Problem.NumCities)

		offspring = append(offspring, child1)
		offspringFits = append(offspringFits, s.Problem.ComputePenalizedFitness(child1, s.Penalty.GetCoefficient()))
		if len(offspring) < popSize {
			offspring = append(offspring, child2)
			offspringFits = append(offspringFits, s.Problem.ComputePenalizedFitness(child2, s.Penalty.GetCoefficient()))
		}
	}

	offspring = offspring[:popSize]
	offspringFits = offspringFits[:popSize]

	for i := 0; i < eliteCount && i < len(elites); i++ {
		worstIdx := 0
		for j := 1; j < len(offspringFits); j++ {
			if offspringFits[j] < offspringFits[worstIdx] {
				worstIdx = j
			}
		}
		if eliteFits[i] > offspringFits[worstIdx] {
			offspring[worstIdx] = elites[i]
			offspringFits[worstIdx] = eliteFits[i]
		}
	}

	s.Population = offspring
	s.Fitnesses = offspringFits
	s.Gen++

	s.updateBest()
}

func (s *TSPTWSolver) updateBest() {
	bestIdx := 0
	for i := 1; i < len(s.Fitnesses); i++ {
		if s.Fitnesses[i] > s.Fitnesses[bestIdx] {
			bestIdx = i
		}
	}

	cost := s.Problem.ComputePenalizedCost(s.Population[bestIdx], s.Penalty.GetCoefficient())
	if s.BestTour == nil || cost < s.BestCost {
		s.BestTour = ensureDepotFirst(s.Population[bestIdx])
		s.BestCost = cost
	}
}

func (s *TSPTWSolver) recomputeAllFitness() {
	for i, tour := range s.Population {
		s.Fitnesses[i] = s.Problem.ComputePenalizedFitness(tour, s.Penalty.GetCoefficient())
	}
	s.updateBest()
}

func (s *TSPTWSolver) recordConvergence() {
	feasibleRatio := ComputeFeasibleRatio(s.Problem, s.Population)
	s.Convergence = append(s.Convergence, ConvergencePoint{
		Generation:    s.Gen,
		BestCost:      s.BestCost,
		FeasibleRatio: feasibleRatio,
	})
}
