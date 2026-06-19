package benchmark

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"strconv"
	"time"

	"github.com/tsp-solver/pkg/config"
	"github.com/tsp-solver/pkg/tsp"
)

type BenchmarkResult struct {
	Instance      string
	Runs          int
	BestDistance  float64
	AvgDistance   float64
	WorstDistance float64
	StdDev        float64
	Optimal       float64
	GapPercent    float64
	AvgTime       time.Duration
}

type GridSearchResult struct {
	PopulationSize int
	Generations    int
	CrossoverRate  float64
	MutationRate   float64
	BestDistance   float64
	AvgDistance    float64
	GapPercent     float64
}

func RunBenchmark(cfg *config.Config) ([]BenchmarkResult, error) {
	var results []BenchmarkResult

	for _, instanceFile := range cfg.Benchmark.Instances {
		problem, err := tsp.LoadTSPProblem(instanceFile, tsp.TSPLIBFormat)
		if err != nil {
			problem, err = tsp.LoadTSPProblem(instanceFile, tsp.Coordinates)
			if err != nil {
				fmt.Printf("Warning: could not load %s: %v\n", instanceFile, err)
				continue
			}
		}

		optimal := 0.0
		if cfg.GA.TSP.OptimalFile != "" {
			optTour, err := tsp.LoadOptimalTour(cfg.GA.TSP.OptimalFile)
			if err == nil && len(optTour) == problem.NumCities {
				optimal = problem.TourLength(optTour)
			}
		}

		distances := make([]float64, cfg.Benchmark.Runs)
		times := make([]time.Duration, cfg.Benchmark.Runs)

		for i := 0; i < cfg.Benchmark.Runs; i++ {
			start := time.Now()

			solver := tsp.NewTSPSolver(problem, &cfg.GA)
			solver.Solve()

			distances[i] = solver.GetBestDistance()
			times[i] = time.Since(start)

			fmt.Printf("Run %d/%d for %s: %.2f\n", i+1, cfg.Benchmark.Runs, instanceFile, distances[i])
		}

		best := math.Inf(1)
		worst := 0.0
		sum := 0.0
		for _, d := range distances {
			if d < best {
				best = d
			}
			if d > worst {
				worst = d
			}
			sum += d
		}
		avg := sum / float64(cfg.Benchmark.Runs)

		variance := 0.0
		for _, d := range distances {
			diff := d - avg
			variance += diff * diff
		}
		stdDev := math.Sqrt(variance / float64(cfg.Benchmark.Runs))

		totalTime := time.Duration(0)
		for _, t := range times {
			totalTime += t
		}
		avgTime := totalTime / time.Duration(cfg.Benchmark.Runs)

		gap := 0.0
		if optimal > 0 {
			gap = (best - optimal) / optimal * 100
		}

		result := BenchmarkResult{
			Instance:      instanceFile,
			Runs:          cfg.Benchmark.Runs,
			BestDistance:  best,
			AvgDistance:   avg,
			WorstDistance: worst,
			StdDev:        stdDev,
			Optimal:       optimal,
			GapPercent:    gap,
			AvgTime:       avgTime,
		}
		results = append(results, result)
	}

	if cfg.Benchmark.OutputFile != "" {
		if err := writeBenchmarkCSV(results, cfg.Benchmark.OutputFile); err != nil {
			fmt.Printf("Warning: could not write benchmark CSV: %v\n", err)
		}
	}

	return results, nil
}

func writeBenchmarkCSV(results []BenchmarkResult, filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	header := []string{
		"instance", "runs", "best_distance", "avg_distance",
		"worst_distance", "std_dev", "optimal", "gap_percent", "avg_time_ms",
	}
	if err := writer.Write(header); err != nil {
		return err
	}

	for _, r := range results {
		row := []string{
			r.Instance,
			strconv.Itoa(r.Runs),
			strconv.FormatFloat(r.BestDistance, 'f', 4, 64),
			strconv.FormatFloat(r.AvgDistance, 'f', 4, 64),
			strconv.FormatFloat(r.WorstDistance, 'f', 4, 64),
			strconv.FormatFloat(r.StdDev, 'f', 4, 64),
			strconv.FormatFloat(r.Optimal, 'f', 4, 64),
			strconv.FormatFloat(r.GapPercent, 'f', 2, 64),
			strconv.FormatFloat(float64(r.AvgTime.Milliseconds()), 'f', 2, 64),
		}
		if err := writer.Write(row); err != nil {
			return err
		}
	}

	return nil
}

func RunGridSearch(baseConfig *config.Config, instanceFile string) ([]GridSearchResult, error) {
	gsCfg := &baseConfig.GridSearch

	problem, err := tsp.LoadTSPProblem(instanceFile, tsp.TSPLIBFormat)
	if err != nil {
		problem, err = tsp.LoadTSPProblem(instanceFile, tsp.Coordinates)
		if err != nil {
			return nil, fmt.Errorf("could not load problem: %v", err)
		}
	}

	optimal := 0.0
	if baseConfig.GA.TSP.OptimalFile != "" {
		optTour, err := tsp.LoadOptimalTour(baseConfig.GA.TSP.OptimalFile)
		if err == nil && len(optTour) == problem.NumCities {
			optimal = problem.TourLength(optTour)
		}
	}

	var results []GridSearchResult

	popSizes := gsCfg.PopulationSizes
	if len(popSizes) == 0 {
		popSizes = []int{baseConfig.GA.PopulationSize}
	}

	gens := gsCfg.Generations
	if len(gens) == 0 {
		gens = []int{baseConfig.GA.Generations}
	}

	crossoverRates := gsCfg.CrossoverRates
	if len(crossoverRates) == 0 {
		crossoverRates = []float64{baseConfig.GA.Crossover.Rate}
	}

	mutationRates := gsCfg.MutationRates
	if len(mutationRates) == 0 {
		mutationRates = []float64{baseConfig.GA.Mutation.Rate}
	}

	totalRuns := len(popSizes) * len(gens) * len(crossoverRates) * len(mutationRates)
	current := 0

	for _, popSize := range popSizes {
		for _, gen := range gens {
			for _, cxRate := range crossoverRates {
				for _, mutRate := range mutationRates {
					current++
					fmt.Printf("Grid search %d/%d: pop=%d, gen=%d, cx=%.2f, mut=%.4f\n",
						current, totalRuns, popSize, gen, cxRate, mutRate)

					testCfg := *baseConfig
					testCfg.GA.PopulationSize = popSize
					testCfg.GA.Generations = gen
					testCfg.GA.Crossover.Rate = cxRate
					testCfg.GA.Mutation.Rate = mutRate

					solver := tsp.NewTSPSolver(problem, &testCfg.GA)
					solver.Solve()

					bestDist := solver.GetBestDistance()
					gap := 0.0
					if optimal > 0 {
						gap = (bestDist - optimal) / optimal * 100
					}

					result := GridSearchResult{
						PopulationSize: popSize,
						Generations:    gen,
						CrossoverRate:  cxRate,
						MutationRate:   mutRate,
						BestDistance:   bestDist,
						AvgDistance:    bestDist,
						GapPercent:     gap,
					}
					results = append(results, result)
				}
			}
		}
	}

	if gsCfg.OutputFile != "" {
		if err := writeGridSearchCSV(results, gsCfg.OutputFile); err != nil {
			fmt.Printf("Warning: could not write grid search CSV: %v\n", err)
		}
	}

	return results, nil
}

func writeGridSearchCSV(results []GridSearchResult, filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	header := []string{
		"population_size", "generations", "crossover_rate",
		"mutation_rate", "best_distance", "gap_percent",
	}
	if err := writer.Write(header); err != nil {
		return err
	}

	for _, r := range results {
		row := []string{
			strconv.Itoa(r.PopulationSize),
			strconv.Itoa(r.Generations),
			strconv.FormatFloat(r.CrossoverRate, 'f', 4, 64),
			strconv.FormatFloat(r.MutationRate, 'f', 6, 64),
			strconv.FormatFloat(r.BestDistance, 'f', 4, 64),
			strconv.FormatFloat(r.GapPercent, 'f', 2, 64),
		}
		if err := writer.Write(row); err != nil {
			return err
		}
	}

	return nil
}
