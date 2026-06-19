package adaptive

import (
	"math"

	"github.com/tsp-solver/pkg/ga/encoding"
)

type AdaptiveConfig struct {
	Enabled        bool
	BaseCrossoverRate float64
	BaseMutationRate  float64
	MinCrossoverRate  float64
	MaxCrossoverRate  float64
	MinMutationRate   float64
	MaxMutationRate   float64
	VarianceThreshold float64
	Sensitivity       float64
}

type AdaptiveRates struct {
	CrossoverRate float64
	MutationRate  float64
}

func CalculateAdaptiveRates(pop encoding.Population, config AdaptiveConfig) AdaptiveRates {
	rates := AdaptiveRates{
		CrossoverRate: config.BaseCrossoverRate,
		MutationRate:  config.BaseMutationRate,
	}

	if !config.Enabled {
		return rates
	}

	variance := pop.FitnessVariance()
	avgFitness := pop.AvgFitness()

	if math.Abs(avgFitness) < 1e-10 {
		return rates
	}

	normalizedVariance := variance / (avgFitness * avgFitness)
	if normalizedVariance < 0 {
		normalizedVariance = 0
	}

	threshold := config.VarianceThreshold
	if threshold <= 0 {
		threshold = 0.01
	}

	sensitivity := config.Sensitivity
	if sensitivity <= 0 {
		sensitivity = 1.0
	}

	ratio := normalizedVariance / threshold
	if ratio > 1 {
		ratio = 1
	}

	variationFactor := 1.0 - ratio
	variationFactor *= sensitivity
	if variationFactor < 0 {
		variationFactor = 0
	}
	if variationFactor > 1 {
		variationFactor = 1
	}

	minCr := config.MinCrossoverRate
	maxCr := config.MaxCrossoverRate
	if minCr <= 0 {
		minCr = 0.3
	}
	if maxCr <= 0 || maxCr > 1 {
		maxCr = 0.95
	}
	if minCr > maxCr {
		minCr, maxCr = maxCr, minCr
	}

	minMr := config.MinMutationRate
	maxMr := config.MaxMutationRate
	if minMr <= 0 {
		minMr = 0.001
	}
	if maxMr <= 0 || maxMr > 1 {
		maxMr = 0.5
	}
	if minMr > maxMr {
		minMr, maxMr = maxMr, minMr
	}

	rates.CrossoverRate = maxCr - variationFactor*(maxCr-minCr)
	rates.MutationRate = minMr + variationFactor*(maxMr-minMr)

	return rates
}
