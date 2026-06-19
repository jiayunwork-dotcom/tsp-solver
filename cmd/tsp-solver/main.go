package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/tsp-solver/pkg/benchmark"
	"github.com/tsp-solver/pkg/config"
	"github.com/tsp-solver/pkg/output"
	"github.com/tsp-solver/pkg/tsp"
	"github.com/tsp-solver/pkg/tsptw"
	"github.com/tsp-solver/pkg/utils"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	command := os.Args[1]

	switch command {
	case "solve":
		runSolve(os.Args[2:])
	case "benchmark":
		runBenchmark(os.Args[2:])
	case "grid-search":
		runGridSearch(os.Args[2:])
	case "visualize":
		runVisualize(os.Args[2:])
	case "tsptw":
		runTSPTW(os.Args[2:])
	case "help", "--help", "-h":
		printUsage()
	default:
		fmt.Printf("Unknown command: %s\n\n", command)
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println("TSP Solver - Genetic Algorithm Framework")
	fmt.Println()
	fmt.Println("Usage:")
	fmt.Println("  tsp-solver <command> [arguments]")
	fmt.Println()
	fmt.Println("Commands:")
	fmt.Println("  solve          Solve a TSP problem using genetic algorithm")
	fmt.Println("  tsptw          Solve a TSP with Time Windows problem")
	fmt.Println("  benchmark      Run benchmark tests on multiple instances")
	fmt.Println("  grid-search    Run grid search for parameter optimization")
	fmt.Println("  visualize      Generate SVG visualization from result file")
	fmt.Println()
	fmt.Println("Use 'tsp-solver <command> --help' for more information about a command.")
}

func runSolve(args []string) {
	fs := flag.NewFlagSet("solve", flag.ExitOnError)
	configFile := fs.String("config", "config.yaml", "Path to YAML configuration file")
	inputFile := fs.String("input", "", "Input file (overrides config)")
	format := fs.String("format", "", "Data format: coordinates, distance_matrix, tsplib (overrides config)")
	outputCSV := fs.String("csv", "", "Output CSV file for convergence data (overrides config)")
	outputSVG := fs.String("svg", "", "Output SVG file for visualization (overrides config)")
	outputResult := fs.String("result", "", "Output result file (overrides config)")
	seed := fs.Int64("seed", 0, "Random seed")
	verbose := fs.Bool("verbose", false, "Verbose output")
	bestKnown := fs.Float64("best-known", 0, "Known optimal solution distance for GAP calculation")

	population := fs.Int("population", 0, "Population size (overrides config)")
	generations := fs.Int("generations", 0, "Number of generations (overrides config)")
	crossoverRate := fs.Float64("crossover-rate", 0, "Crossover rate (overrides config)")
	mutationRate := fs.Float64("mutation-rate", 0, "Mutation rate (overrides config)")
	tournamentSize := fs.Int("tournament-size", 0, "Tournament size (overrides config)")
	eliteCount := fs.Int("elite-count", 0, "Elite count (overrides config)")
	localSearch := fs.String("local-search", "", "Local search type: 2-opt, 3-opt, or-opt (overrides config)")
	useLocalSearch := fs.Bool("use-local-search", false, "Enable local search (overrides config)")

	fs.Parse(args)

	cfg, err := config.LoadConfig(*configFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
		os.Exit(1)
	}

	if *inputFile != "" {
		cfg.GA.TSP.InputFile = *inputFile
	}
	if *format != "" {
		cfg.GA.TSP.DataFormat = *format
	}
	if *outputCSV != "" {
		cfg.GA.Output.CSVFile = *outputCSV
	}
	if *outputSVG != "" {
		cfg.GA.Output.SVGFile = *outputSVG
	}
	if *outputResult != "" {
		cfg.GA.Output.ResultFile = *outputResult
	}
	if *verbose {
		cfg.GA.Output.Verbose = true
	}

	if *seed > 0 {
		utils.SetSeed(*seed)
	}

	if *population > 0 {
		cfg.GA.PopulationSize = *population
	}
	if *generations > 0 {
		cfg.GA.Generations = *generations
	}
	if *crossoverRate > 0 {
		cfg.GA.Crossover.Rate = *crossoverRate
	}
	if *mutationRate > 0 {
		cfg.GA.Mutation.Rate = *mutationRate
	}
	if *tournamentSize > 0 {
		cfg.GA.Selection.TournamentSize = *tournamentSize
	}
	if *eliteCount > 0 {
		cfg.GA.Selection.EliteCount = *eliteCount
	}
	if *localSearch != "" {
		cfg.GA.LocalSearch.Type = *localSearch
	}
	if *useLocalSearch {
		cfg.GA.LocalSearch.Enabled = true
	}

	applyOverrides(cfg, args)

	dataFormat := tsp.DataFormat(cfg.GA.TSP.DataFormat)
	problem, err := tsp.LoadTSPProblem(cfg.GA.TSP.InputFile, dataFormat)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading problem: %v\n", err)
		os.Exit(1)
	}

	if cfg.GA.PermutationSize == 0 {
		cfg.GA.PermutationSize = problem.NumCities
	}
	if cfg.GA.GenomeLength == 0 {
		cfg.GA.GenomeLength = problem.NumCities
	}

	fmt.Printf("Problem: %s\n", problem.Name)
	fmt.Printf("Number of cities: %d\n", problem.NumCities)

	solver := tsp.NewTSPSolver(problem, &cfg.GA)
	result := solver.Solve()

	bestTour := solver.GetBestTour()
	bestDistance := solver.GetBestDistance()

	totalGens := result.History.Generations[len(result.History.Generations)-1]
	durationMs := result.Duration.Milliseconds()
	avgGenMs := 0.0
	if totalGens > 0 {
		avgGenMs = float64(durationMs) / float64(totalGens)
	}

	diversityDecay := 0.0
	if result.InitialDiversity > 1e-15 {
		diversityDecay = (result.InitialDiversity - result.FinalDiversity) / result.InitialDiversity * 100.0
	}

	fmt.Println()
	fmt.Println("=== Results ===")
	fmt.Printf("Best tour length: %.4f\n", bestDistance)
	fmt.Printf("Generations: %d\n", totalGens)
	fmt.Printf("Duration: %v\n", result.Duration)

	if len(bestTour) <= 50 {
		fmt.Printf("Best tour: %v\n", bestTour)
	}

	if *bestKnown > 0 {
		gap := (bestDistance - *bestKnown) / *bestKnown * 100
		fmt.Printf("Best known: %.4f\n", *bestKnown)
		fmt.Printf("Gap to best known: %.2f%%\n", gap)
	} else if cfg.GA.TSP.OptimalFile != "" {
		optTour, err := tsp.LoadOptimalTour(cfg.GA.TSP.OptimalFile)
		if err == nil && len(optTour) == problem.NumCities {
			optDist := problem.TourLength(optTour)
			gap := (bestDistance - optDist) / optDist * 100
			fmt.Printf("Optimal distance: %.4f\n", optDist)
			fmt.Printf("Gap to optimal: %.2f%%\n", gap)
		}
	}

	fmt.Println()
	fmt.Println("=== Statistics Summary ===")
	fmt.Printf("Total time: %d ms\n", durationMs)
	fmt.Printf("Avg time per generation: %.2f ms\n", avgGenMs)
	fmt.Printf("First found best at generation: %d\n", result.FirstBestGeneration)
	fmt.Printf("Diversity decay: %.2f%% (initial: %.6f -> final: %.6f)\n", diversityDecay, result.InitialDiversity, result.FinalDiversity)
	if cfg.GA.LocalSearch.Enabled {
		fmt.Printf("Local search calls: %d\n", result.LocalSearchCalls)
		fmt.Printf("Local search total improvement: %.4f\n", result.LocalSearchImproved)
	}
	if len(result.IslandMigrationStats) > 0 {
		fmt.Printf("Best solution from Island #%d\n", result.BestIslandID)
		if *verbose || cfg.GA.Output.Verbose {
			fmt.Println()
			fmt.Println("=== Island Migration Statistics ===")
			for _, stat := range result.IslandMigrationStats {
				improveRatio := 0.0
				if stat.ReceivedMigrations > 0 {
					improveRatio = float64(stat.ImprovedAfter) / float64(stat.ReceivedMigrations) * 100.0
				}
				fmt.Printf("Island #%d: received %d migrations, improved after %d (%.1f%%)\n",
					stat.IslandID, stat.ReceivedMigrations, stat.ImprovedAfter, improveRatio)
			}
		}
	}

	if cfg.GA.Output.CSVFile != "" {
		if err := output.WriteConvergenceCSV(result.History, cfg.GA.Output.CSVFile); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: could not write CSV: %v\n", err)
		} else {
			fmt.Printf("Convergence data written to: %s\n", cfg.GA.Output.CSVFile)
		}
	}

	if cfg.GA.Output.SVGFile != "" {
		if err := output.GenerateTSPVisualization(problem, bestTour, cfg.GA.Output.SVGFile); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: could not generate SVG: %v\n", err)
		} else {
			fmt.Printf("Visualization written to: %s\n", cfg.GA.Output.SVGFile)
		}
	}

	if cfg.GA.Output.ResultFile != "" {
		resultContent := fmt.Sprintf(
			"Problem: %s\nCities: %d\nBest Distance: %.4f\nDuration: %v\n\nTour:\n%v\n",
			problem.Name, problem.NumCities, bestDistance, result.Duration, bestTour)
		if err := os.WriteFile(cfg.GA.Output.ResultFile, []byte(resultContent), 0644); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: could not write result: %v\n", err)
		}
	}
}

func runBenchmark(args []string) {
	fs := flag.NewFlagSet("benchmark", flag.ExitOnError)
	configFile := fs.String("config", "config.yaml", "Path to YAML configuration file")
	instances := fs.String("instances", "", "Comma-separated list of instance files (overrides config)")
	runs := fs.Int("runs", 0, "Number of runs per instance (overrides config)")
	outputFile := fs.String("output", "", "Output CSV file (overrides config)")
	bestKnown := fs.Float64("best-known", 0, "Known optimal solution distance for GAP calculation")

	fs.Parse(args)

	cfg, err := config.LoadConfig(*configFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
		os.Exit(1)
	}

	if *instances != "" {
		cfg.Benchmark.Instances = strings.Split(*instances, ",")
	}
	if *runs > 0 {
		cfg.Benchmark.Runs = *runs
	}
	if *outputFile != "" {
		cfg.Benchmark.OutputFile = *outputFile
	}

	if len(cfg.Benchmark.Instances) == 0 {
		fmt.Fprintln(os.Stderr, "Error: no benchmark instances specified")
		os.Exit(1)
	}

	fmt.Println("Running benchmarks...")
	results, err := benchmark.RunBenchmark(cfg, *bestKnown)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error running benchmark: %v\n", err)
		os.Exit(1)
	}

	fmt.Println()
	fmt.Println("=== Benchmark Results ===")
	for _, r := range results {
		fmt.Printf("\nInstance: %s\n", r.Instance)
		fmt.Printf("  Runs: %d\n", r.Runs)
		fmt.Printf("  Best: %.4f\n", r.BestDistance)
		fmt.Printf("  Avg:  %.4f\n", r.AvgDistance)
		fmt.Printf("  Worst: %.4f\n", r.WorstDistance)
		fmt.Printf("  StdDev: %.4f\n", r.StdDev)
		fmt.Printf("  Avg Time: %v\n", r.AvgTime)
		if r.Optimal > 0 {
			fmt.Printf("  Optimal/Best-known: %.4f\n", r.Optimal)
			fmt.Printf("  Gap: %.2f%%\n", r.GapPercent)
		}
	}
}

func runGridSearch(args []string) {
	fs := flag.NewFlagSet("grid-search", flag.ExitOnError)
	configFile := fs.String("config", "config.yaml", "Path to YAML configuration file")
	inputFile := fs.String("input", "", "Input problem file")
	bestKnown := fs.Float64("best-known", 0, "Known optimal solution distance for GAP calculation")

	fs.Parse(args)

	cfg, err := config.LoadConfig(*configFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
		os.Exit(1)
	}

	if *inputFile != "" {
		cfg.GA.TSP.InputFile = *inputFile
	}

	if cfg.GA.TSP.InputFile == "" {
		fmt.Fprintln(os.Stderr, "Error: no input file specified")
		os.Exit(1)
	}

	fmt.Println("Running grid search...")
	results, err := benchmark.RunGridSearch(cfg, cfg.GA.TSP.InputFile, *bestKnown)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error running grid search: %v\n", err)
		os.Exit(1)
	}

	fmt.Println()
	fmt.Println("=== Grid Search Results ===")
	if *bestKnown > 0 {
		fmt.Printf("(sorted by GAP%%, best-known=%.4f)\n", *bestKnown)
	}
	fmt.Println(strings.Repeat("=", 80))
	fmt.Printf("%-6s %-6s %-8s %-10s %-14s %-10s\n",
		"POP", "GEN", "CX_RATE", "MUT_RATE", "BEST_DIST", "GAP%")
	fmt.Println(strings.Repeat("-", 80))

	for _, r := range results {
		fmt.Printf("%-6d %-6d %-8.4f %-10.6f %-14.2f %-10.2f\n",
			r.PopulationSize, r.Generations, r.CrossoverRate,
			r.MutationRate, r.BestDistance, r.GapPercent)
	}
}

func runVisualize(args []string) {
	fs := flag.NewFlagSet("visualize", flag.ExitOnError)
	inputFile := fs.String("input", "", "Input TSP problem file")
	format := fs.String("format", "coordinates", "Data format: coordinates, distance_matrix, tsplib")
	tourFile := fs.String("tour", "", "File containing the tour (one city per line)")
	outputFile := fs.String("output", "tsp_solution.svg", "Output SVG file")

	fs.Parse(args)

	if *inputFile == "" {
		fmt.Fprintln(os.Stderr, "Error: input file required")
		os.Exit(1)
	}

	problem, err := tsp.LoadTSPProblem(*inputFile, tsp.DataFormat(*format))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading problem: %v\n", err)
		os.Exit(1)
	}

	var tour []int
	if *tourFile != "" {
		tourData, err := os.ReadFile(*tourFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading tour file: %v\n", err)
			os.Exit(1)
		}
		lines := strings.Split(strings.TrimSpace(string(tourData)), "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			city, err := strconv.Atoi(line)
			if err != nil {
				continue
			}
			tour = append(tour, city)
		}
	} else {
		tour = make([]int, problem.NumCities)
		for i := range tour {
			tour[i] = i
		}
	}

	if len(tour) != problem.NumCities {
		fmt.Fprintf(os.Stderr, "Warning: tour length (%d) does not match number of cities (%d)\n",
			len(tour), problem.NumCities)
	}

	distance := problem.TourLength(tour)
	fmt.Printf("Tour length: %.4f\n", distance)

	if err := output.GenerateTSPVisualization(problem, tour, *outputFile); err != nil {
		fmt.Fprintf(os.Stderr, "Error generating SVG: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Visualization written to: %s\n", *outputFile)
}

func runTSPTW(args []string) {
	fs := flag.NewFlagSet("tsptw", flag.ExitOnError)
	configFile := fs.String("config", "config.yaml", "Path to YAML configuration file")
	inputFile := fs.String("input", "", "Input TSPTW data file (overrides config)")
	outputSVG := fs.String("svg", "", "Output SVG file (overrides config)")
	outputResult := fs.String("result", "", "Output result file (overrides config)")
	seed := fs.Int64("seed", 0, "Random seed")
	verbose := fs.Bool("verbose", false, "Verbose output")
	compareRandom := fs.Int("compare-random", 0, "Compare with N random tours")
	exportJSON := fs.String("export-json", "", "Export results to JSON file")

	population := fs.Int("population", 0, "Population size (overrides config)")
	generations := fs.Int("generations", 0, "Number of generations (overrides config)")
	penaltyType := fs.String("penalty-type", "", "Penalty type: fixed, adaptive (overrides config)")
	penaltyCoeff := fs.Float64("penalty-coefficient", 0, "Penalty coefficient (overrides config)")
	speed := fs.Float64("speed", 0, "Travel speed (overrides config)")
	repairEnabled := fs.Bool("repair", false, "Enable repair operator (overrides config)")

	fs.Parse(args)

	cfg, err := config.LoadConfig(*configFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
		os.Exit(1)
	}

	if *inputFile != "" {
		cfg.TSPTW.InputFile = *inputFile
	}
	if *outputSVG != "" {
		cfg.TSPTW.OutputSVG = *outputSVG
	}
	if *outputResult != "" {
		cfg.TSPTW.OutputResult = *outputResult
	}
	if *verbose {
		cfg.TSPTW.Verbose = true
	}
	if *seed > 0 {
		utils.SetSeed(*seed)
	}
	if *population > 0 {
		cfg.TSPTW.PopulationSize = *population
	}
	if *generations > 0 {
		cfg.TSPTW.Generations = *generations
	}
	if *penaltyType != "" {
		cfg.TSPTW.PenaltyType = *penaltyType
	}
	if *penaltyCoeff > 0 {
		cfg.TSPTW.PenaltyCoefficient = *penaltyCoeff
	}
	if *speed > 0 {
		cfg.TSPTW.Speed = *speed
	}
	if *repairEnabled {
		cfg.TSPTW.RepairEnabled = true
	}

	if cfg.TSPTW.InputFile == "" {
		fmt.Fprintln(os.Stderr, "Error: no input file specified (use --input or set tsptw.input_file in config)")
		os.Exit(1)
	}

	problem, err := tsptw.LoadTSPTWProblem(cfg.TSPTW.InputFile, cfg.TSPTW.Speed)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading TSPTW problem: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Problem: %s\n", problem.Name)
	fmt.Printf("Number of cities: %d\n", problem.NumCities)
	fmt.Printf("Speed: %.2f\n", problem.Speed)
	fmt.Println()
	fmt.Println("=== City Time Windows ===")
	for _, c := range problem.Cities {
		fmt.Printf("  City %d: (%.1f, %.1f) TW=[%.0f, %.0f] Service=%.0f\n",
			c.ID, c.X, c.Y, c.Earliest, c.Latest, c.ServiceTime)
	}

	solver := tsptw.NewTSPTWSolver(problem, &cfg.TSPTW)
	result := solver.Solve()

	eval := result.BestEval

	fmt.Println()
	fmt.Println("=== TSPTW Results ===")
	fmt.Printf("Best tour: %v\n", result.BestTour)
	fmt.Printf("Total travel distance: %.4f\n", eval.TotalDistance)
	fmt.Printf("Total wait time: %.4f\n", eval.TotalWaitTime)
	fmt.Printf("Total violation time: %.4f\n", eval.TotalViolation)
	fmt.Printf("Feasible: %v\n", eval.IsFeasible)
	fmt.Printf("Penalized cost: %.4f\n", result.BestCost)
	fmt.Printf("Final penalty coefficient: %.4f\n", result.FinalPenalty)
	fmt.Printf("Feasible ratio in population: %.2f%%\n", result.FeasibleRatio*100)
	fmt.Printf("Duration: %v\n", result.Duration)

	fmt.Println()
	fmt.Println("=== Time Schedule ===")
	fmt.Printf("%-18s %-8s %-8s %-8s %-8s %-10s %-10s\n",
		"City", "Arrive", "Wait", "Start", "Leave", "TW", "Violation")
	fmt.Println(strings.Repeat("-", 80))
	for _, v := range eval.Visits {
		city := problem.Cities[v.CityID]
		violStr := ""
		if v.Violation > 1e-10 {
			violStr = fmt.Sprintf("*** %.2f", v.Violation)
		}
		cityLabel := fmt.Sprintf("%d", v.CityID)
		if v.CityID == 0 {
			cityLabel = "0 (Depot)"
		}
		fmt.Printf("%-18s %-8.2f %-8.2f %-8.2f %-8.2f [%-.0f,%-.0f]  %-10s\n",
			cityLabel, v.ArrivalTime, v.WaitTime, v.ServiceStart, v.ServiceEnd,
			city.Earliest, city.Latest, violStr)
	}
	returnViolStr := ""
	if eval.ReturnViolation > 1e-10 {
		returnViolStr = fmt.Sprintf("*** %.2f", eval.ReturnViolation)
	}
	depotLatest := problem.Cities[0].Latest
	fmt.Printf("%-18s %-8.2f %-8s %-8s %-8s [%-.0f,%-.0f]  %-10s\n",
		"Return to depot", eval.ReturnArrivalTime, "-", "-", "-",
		0.0, depotLatest, returnViolStr)
	fmt.Println(strings.Repeat("-", 80))
	fmt.Printf("Total wait: %.2f, Total violation: %.2f\n", eval.TotalWaitTime, eval.TotalViolation)

	if !eval.IsFeasible {
		fmt.Println()
		fmt.Println("!!! TIME WINDOW VIOLATIONS DETECTED !!!")
		for _, v := range eval.Visits {
			if v.Violation > 1e-10 {
				fmt.Printf("  City %d: arrived at %.2f, latest = %.0f, VIOLATED by %.2f minutes\n",
					v.CityID, v.ServiceStart, problem.Cities[v.CityID].Latest, v.Violation)
			}
		}
		if eval.ReturnViolation > 1e-10 {
			fmt.Printf("  Return to depot: arrived at %.2f, latest = %.0f, VIOLATED by %.2f minutes\n",
				eval.ReturnArrivalTime, problem.Cities[0].Latest, eval.ReturnViolation)
		}
	}

	if cfg.TSPTW.OutputSVG != "" {
		if err := output.GenerateTSPTWVisualization(problem, result.BestTour, eval, cfg.TSPTW.OutputSVG); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: could not generate SVG: %v\n", err)
		} else {
			fmt.Printf("Visualization written to: %s\n", cfg.TSPTW.OutputSVG)
		}
	}

	if cfg.TSPTW.OutputResult != "" {
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("Problem: %s\nCities: %d\nBest Distance: %.4f\nTotal Wait: %.4f\nTotal Violation: %.4f\nFeasible: %v\nDuration: %v\n\nTour:\n%v\n\n",
			problem.Name, problem.NumCities, eval.TotalDistance, eval.TotalWaitTime, eval.TotalViolation, eval.IsFeasible, result.Duration, result.BestTour))
		sb.WriteString("Time Schedule:\n")
		for _, v := range eval.Visits {
			city := problem.Cities[v.CityID]
			violStr := ""
			if v.Violation > 1e-10 {
				violStr = fmt.Sprintf(" VIOLATED(%.2f)", v.Violation)
			}
			sb.WriteString(fmt.Sprintf("City %d: arrive=%.2f wait=%.2f start=%.2f leave=%.2f TW=[%.0f,%.0f]%s\n",
				v.CityID, v.ArrivalTime, v.WaitTime, v.ServiceStart, v.ServiceEnd, city.Earliest, city.Latest, violStr))
		}
		returnViolStr := ""
		if eval.ReturnViolation > 1e-10 {
			returnViolStr = fmt.Sprintf(" VIOLATED(%.2f)", eval.ReturnViolation)
		}
		sb.WriteString(fmt.Sprintf("Return to depot: arrive=%.2f TW=[%.0f,%.0f]%s\n",
			eval.ReturnArrivalTime, 0.0, problem.Cities[0].Latest, returnViolStr))
		if err := os.WriteFile(cfg.TSPTW.OutputResult, []byte(sb.String()), 0644); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: could not write result: %v\n", err)
		} else {
			fmt.Printf("Result written to: %s\n", cfg.TSPTW.OutputResult)
		}
	}

	if *compareRandom > 0 {
		fmt.Println()
		fmt.Println("=== Random Solution Comparison ===")
		fmt.Printf("Generating %d random tours...\n", *compareRandom)

		bestDist := math.Inf(1)
		worstDist := 0.0
		bestWait := 0.0
		worstWait := 0.0
		bestViolation := 0.0
		worstViolation := 0.0
		totalDist := 0.0
		totalWait := 0.0
		totalViolation := 0.0
		feasibleCount := 0
		bestFeasible := false
		worstFeasible := false

		for i := 0; i < *compareRandom; i++ {
			randomTour := tsptw.RandomTour(problem.NumCities)
			randomEval := problem.EvaluateTour(randomTour)

			totalDist += randomEval.TotalDistance
			totalWait += randomEval.TotalWaitTime
			totalViolation += randomEval.TotalViolation

			if randomEval.IsFeasible {
				feasibleCount++
			}

			if randomEval.TotalDistance < bestDist {
				bestDist = randomEval.TotalDistance
				bestWait = randomEval.TotalWaitTime
				bestViolation = randomEval.TotalViolation
				bestFeasible = randomEval.IsFeasible
			}
			if randomEval.TotalDistance > worstDist {
				worstDist = randomEval.TotalDistance
				worstWait = randomEval.TotalWaitTime
				worstViolation = randomEval.TotalViolation
				worstFeasible = randomEval.IsFeasible
			}
		}

		avgDist := totalDist / float64(*compareRandom)
		avgWait := totalWait / float64(*compareRandom)
		avgViolation := totalViolation / float64(*compareRandom)
		avgFeasible := float64(feasibleCount) / float64(*compareRandom)

		fmt.Println()
		fmt.Println(strings.Repeat("=", 90))
		fmt.Printf("%-20s %-15s %-15s %-15s %-15s\n",
			"Method", "Total Distance", "Total Wait", "Total Violation", "Feasible")
		fmt.Println(strings.Repeat("-", 90))
		fmt.Printf("%-20s %-15.2f %-15.2f %-15.2f %-15v\n",
			"GA Best", eval.TotalDistance, eval.TotalWaitTime, eval.TotalViolation, eval.IsFeasible)
		fmt.Printf("%-20s %-15.2f %-15.2f %-15.2f %-15v\n",
			"Random Best", bestDist, bestWait, bestViolation, bestFeasible)
		fmt.Printf("%-20s %-15.2f %-15.2f %-15.2f %-15.2f%%\n",
			"Random Avg", avgDist, avgWait, avgViolation, avgFeasible*100)
		fmt.Printf("%-20s %-15.2f %-15.2f %-15.2f %-15v\n",
			"Random Worst", worstDist, worstWait, worstViolation, worstFeasible)
		fmt.Println(strings.Repeat("=", 90))

		gapBest := 0.0
		if bestDist > 1e-10 {
			gapBest = (eval.TotalDistance - bestDist) / bestDist * 100
		}
		gapAvg := 0.0
		if avgDist > 1e-10 {
			gapAvg = (avgDist - eval.TotalDistance) / avgDist * 100
		}
		fmt.Printf("\nGA vs Random Best:   %+.2f%%\n", gapBest)
		fmt.Printf("GA vs Random Avg:    %+.2f%% improvement\n", gapAvg)
		fmt.Printf("Random feasible rate: %.2f%%\n", avgFeasible*100)
	}

	if *exportJSON != "" {
		err := exportTSPTWJSON(problem, result, eval, *exportJSON)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: could not export JSON: %v\n", err)
		} else {
			fmt.Printf("JSON exported to: %s\n", *exportJSON)
		}
	}
}

type jsonScheduleEntry struct {
	CityID    int     `json:"city_id"`
	Arrive    float64 `json:"arrive"`
	Wait      float64 `json:"wait"`
	Start     float64 `json:"start"`
	Leave     float64 `json:"leave"`
	Violation float64 `json:"violation"`
}

type jsonReturnDepot struct {
	Arrive    float64 `json:"arrive"`
	Violation float64 `json:"violation"`
}

type jsonSummary struct {
	Distance   float64 `json:"distance"`
	Wait       float64 `json:"wait"`
	Violation  float64 `json:"violation"`
	Feasible   bool    `json:"feasible"`
	DurationMs int64   `json:"duration_ms"`
	Generations int    `json:"generations"`
}

type jsonConvergencePoint struct {
	Generation    int     `json:"generation"`
	BestCost      float64 `json:"best_cost"`
	FeasibleRatio float64 `json:"feasible_ratio"`
}

type jsonTSPTWResult struct {
	Tour        []int                   `json:"tour"`
	Schedule    []jsonScheduleEntry     `json:"schedule"`
	ReturnDepot jsonReturnDepot         `json:"return_depot"`
	Summary     jsonSummary             `json:"summary"`
	Convergence []jsonConvergencePoint  `json:"convergence"`
}

func exportTSPTWJSON(problem *tsptw.TSPTWProblem, result *tsptw.TSPTWResult, eval *tsptw.TourEvaluation, filePath string) error {
	schedule := make([]jsonScheduleEntry, len(eval.Visits))
	for i, v := range eval.Visits {
		schedule[i] = jsonScheduleEntry{
			CityID:    v.CityID,
			Arrive:    v.ArrivalTime,
			Wait:      v.WaitTime,
			Start:     v.ServiceStart,
			Leave:     v.ServiceEnd,
			Violation: v.Violation,
		}
	}

	conv := make([]jsonConvergencePoint, len(result.Convergence))
	for i, c := range result.Convergence {
		conv[i] = jsonConvergencePoint{
			Generation:    c.Generation,
			BestCost:      c.BestCost,
			FeasibleRatio: c.FeasibleRatio,
		}
	}

	jsonResult := jsonTSPTWResult{
		Tour:     result.BestTour,
		Schedule: schedule,
		ReturnDepot: jsonReturnDepot{
			Arrive:    eval.ReturnArrivalTime,
			Violation: eval.ReturnViolation,
		},
		Summary: jsonSummary{
			Distance:    eval.TotalDistance,
			Wait:        eval.TotalWaitTime,
			Violation:   eval.TotalViolation,
			Feasible:    eval.IsFeasible,
			DurationMs:  result.Duration.Milliseconds(),
			Generations: result.Generations,
		},
		Convergence: conv,
	}

	data, err := json.MarshalIndent(jsonResult, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %v", err)
	}

	if err := os.WriteFile(filePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write JSON file: %v", err)
	}

	return nil
}

func applyOverrides(cfg *config.Config, args []string) {
	for _, arg := range args {
		if !strings.HasPrefix(arg, "--") {
			continue
		}
		arg = strings.TrimPrefix(arg, "--")
		parts := strings.SplitN(arg, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := parts[0]
		value := parts[1]

		switch key {
		case "population":
			if v, err := strconv.Atoi(value); err == nil {
				cfg.GA.PopulationSize = v
			}
		case "generations":
			if v, err := strconv.Atoi(value); err == nil {
				cfg.GA.Generations = v
			}
		case "crossover-rate":
			if v, err := strconv.ParseFloat(value, 64); err == nil {
				cfg.GA.Crossover.Rate = v
			}
		case "mutation-rate":
			if v, err := strconv.ParseFloat(value, 64); err == nil {
				cfg.GA.Mutation.Rate = v
			}
		case "tournament-size":
			if v, err := strconv.Atoi(value); err == nil {
				cfg.GA.Selection.TournamentSize = v
			}
		case "elite-count":
			if v, err := strconv.Atoi(value); err == nil {
				cfg.GA.Selection.EliteCount = v
			}
		}
	}
}
