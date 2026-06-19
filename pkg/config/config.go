package config

import (
	"fmt"
	"io/ioutil"
	"os"

	"gopkg.in/yaml.v3"
)

type SelectionConfig struct {
	Type           string `yaml:"type"`
	TournamentSize int    `yaml:"tournament_size"`
	EliteCount     int    `yaml:"elite_count"`
}

type CrossoverConfig struct {
	Type    string  `yaml:"type"`
	Rate    float64 `yaml:"rate"`
	SBX_Eta float64 `yaml:"sbx_eta"`
}

type MutationConfig struct {
	Type        string  `yaml:"type"`
	Rate        float64 `yaml:"rate"`
	GaussianStd float64 `yaml:"gaussian_std"`
}

type AdaptiveConfig struct {
	Enabled           bool    `yaml:"enabled"`
	BaseCrossoverRate float64 `yaml:"base_crossover_rate"`
	BaseMutationRate  float64 `yaml:"base_mutation_rate"`
	MinCrossoverRate  float64 `yaml:"min_crossover_rate"`
	MaxCrossoverRate  float64 `yaml:"max_crossover_rate"`
	MinMutationRate   float64 `yaml:"min_mutation_rate"`
	MaxMutationRate   float64 `yaml:"max_mutation_rate"`
	VarianceThreshold float64 `yaml:"variance_threshold"`
	Sensitivity       float64 `yaml:"sensitivity"`
}

type LocalSearchConfig struct {
	Enabled        bool   `yaml:"enabled"`
	Type           string `yaml:"type"`
	Interval       int    `yaml:"interval"`
	TopK           int    `yaml:"top_k"`
	UseKDTree      bool   `yaml:"use_kdtree"`
	KDTreeNeighbors int   `yaml:"kdtree_neighbors"`
	KDTreeThreshold int   `yaml:"kdtree_threshold"`
}

type IslandGAConfig struct {
	PopulationSize  int             `yaml:"population_size"`
	CrossoverRate   float64         `yaml:"crossover_rate"`
	MutationRate    float64         `yaml:"mutation_rate"`
	SelectionType   string          `yaml:"selection_type"`
	CrossoverType   string          `yaml:"crossover_type"`
	MutationType    string          `yaml:"mutation_type"`
}

type IslandConfig struct {
	Enabled      bool            `yaml:"enabled"`
	NumIslands   int             `yaml:"num_islands"`
	Migration    MigrationConfig `yaml:"migration"`
	IslandConfigs []IslandGAConfig `yaml:"island_configs"`
}

type MigrationConfig struct {
	Interval    int    `yaml:"interval"`
	NumMigrants int    `yaml:"num_migrants"`
	Topology    string `yaml:"topology"`
}

type TSPConfig struct {
	DataFormat string `yaml:"data_format"`
	InputFile  string `yaml:"input_file"`
	OptimalFile string `yaml:"optimal_file"`
}

type NSGA2Config struct {
	Enabled       bool    `yaml:"enabled"`
	NumObjectives int     `yaml:"num_objectives"`
}

type OutputConfig struct {
	CSVFile       string `yaml:"csv_file"`
	SVGFile       string `yaml:"svg_file"`
	ResultFile    string `yaml:"result_file"`
	Verbose       bool   `yaml:"verbose"`
}

type GAConfig struct {
	Encoding        string          `yaml:"encoding"`
	PopulationSize  int             `yaml:"population_size"`
	Generations     int             `yaml:"generations"`
	GenomeLength    int             `yaml:"genome_length"`
	PermutationSize int             `yaml:"permutation_size"`

	Selection SelectionConfig `yaml:"selection"`
	Crossover CrossoverConfig `yaml:"crossover"`
	Mutation  MutationConfig  `yaml:"mutation"`
	Adaptive  AdaptiveConfig  `yaml:"adaptive"`

	LocalSearch LocalSearchConfig `yaml:"local_search"`
	Island      IslandConfig      `yaml:"island"`
	NSGA2       NSGA2Config       `yaml:"nsga2"`

	TSP    TSPConfig    `yaml:"tsp"`
	Output OutputConfig `yaml:"output"`
}

type GridSearchConfig struct {
	PopulationSizes  []int     `yaml:"population_sizes"`
	Generations      []int     `yaml:"generations"`
	CrossoverRates   []float64 `yaml:"crossover_rates"`
	MutationRates    []float64 `yaml:"mutation_rates"`
	OutputFile       string    `yaml:"output_file"`
}

type BenchmarkConfig struct {
	Instances []string `yaml:"instances"`
	Runs      int      `yaml:"runs"`
	OutputFile string  `yaml:"output_file"`
}

type TSPTWConfig struct {
	PenaltyType           string  `yaml:"penalty_type"`
	PenaltyCoefficient    float64 `yaml:"penalty_coefficient"`
	FeasibilityTarget     float64 `yaml:"feasibility_target"`
	PenaltyAdjustInterval int     `yaml:"penalty_adjust_interval"`
	RepairEnabled         bool    `yaml:"repair_enabled"`
	RepairInterval        int     `yaml:"repair_interval"`
	RepairTopK            int     `yaml:"repair_top_k"`
	Speed                 float64 `yaml:"speed"`
	InputFile             string  `yaml:"input_file"`
	OutputCSV             string  `yaml:"output_csv"`
	OutputSVG             string  `yaml:"output_svg"`
	OutputResult          string  `yaml:"output_result"`
	PopulationSize        int     `yaml:"population_size"`
	Generations           int     `yaml:"generations"`
	CrossoverRate         float64 `yaml:"crossover_rate"`
	MutationRate          float64 `yaml:"mutation_rate"`
	TournamentSize        int     `yaml:"tournament_size"`
	EliteCount            int     `yaml:"elite_count"`
	Verbose               bool    `yaml:"verbose"`
}

type Config struct {
	GA         GAConfig         `yaml:"ga"`
	GridSearch GridSearchConfig `yaml:"grid_search"`
	Benchmark  BenchmarkConfig  `yaml:"benchmark"`
	TSPTW      TSPTWConfig      `yaml:"tsptw"`
}

func LoadConfig(filePath string) (*Config, error) {
	data, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %v", err)
	}

	var config Config
	err = yaml.Unmarshal(data, &config)
	if err != nil {
		return nil, fmt.Errorf("failed to parse config file: %v", err)
	}

	setDefaults(&config)

	return &config, nil
}

func setDefaults(config *Config) {
	if config.GA.Encoding == "" {
		config.GA.Encoding = "permutation"
	}
	if config.GA.PopulationSize == 0 {
		config.GA.PopulationSize = 100
	}
	if config.GA.Generations == 0 {
		config.GA.Generations = 1000
	}

	if config.GA.Selection.Type == "" {
		config.GA.Selection.Type = "tournament"
	}
	if config.GA.Selection.TournamentSize == 0 {
		config.GA.Selection.TournamentSize = 3
	}
	if config.GA.Selection.EliteCount == 0 {
		config.GA.Selection.EliteCount = 2
	}

	if config.GA.Crossover.Type == "" {
		config.GA.Crossover.Type = "ox"
	}
	if config.GA.Crossover.Rate == 0 {
		config.GA.Crossover.Rate = 0.8
	}
	if config.GA.Crossover.SBX_Eta == 0 {
		config.GA.Crossover.SBX_Eta = 20
	}

	if config.GA.Mutation.Type == "" {
		config.GA.Mutation.Type = "swap"
	}
	if config.GA.Mutation.Rate == 0 {
		config.GA.Mutation.Rate = 0.02
	}
	if config.GA.Mutation.GaussianStd == 0 {
		config.GA.Mutation.GaussianStd = 0.1
	}

	if config.GA.Adaptive.BaseCrossoverRate == 0 {
		config.GA.Adaptive.BaseCrossoverRate = config.GA.Crossover.Rate
	}
	if config.GA.Adaptive.BaseMutationRate == 0 {
		config.GA.Adaptive.BaseMutationRate = config.GA.Mutation.Rate
	}
	if config.GA.Adaptive.MinCrossoverRate == 0 {
		config.GA.Adaptive.MinCrossoverRate = 0.3
	}
	if config.GA.Adaptive.MaxCrossoverRate == 0 {
		config.GA.Adaptive.MaxCrossoverRate = 0.95
	}
	if config.GA.Adaptive.MinMutationRate == 0 {
		config.GA.Adaptive.MinMutationRate = 0.001
	}
	if config.GA.Adaptive.MaxMutationRate == 0 {
		config.GA.Adaptive.MaxMutationRate = 0.5
	}
	if config.GA.Adaptive.VarianceThreshold == 0 {
		config.GA.Adaptive.VarianceThreshold = 0.01
	}
	if config.GA.Adaptive.Sensitivity == 0 {
		config.GA.Adaptive.Sensitivity = 1.0
	}

	if config.GA.LocalSearch.Type == "" {
		config.GA.LocalSearch.Type = "2-opt"
	}
	if config.GA.LocalSearch.Interval == 0 {
		config.GA.LocalSearch.Interval = 50
	}
	if config.GA.LocalSearch.TopK == 0 {
		config.GA.LocalSearch.TopK = 10
	}
	if config.GA.LocalSearch.KDTreeNeighbors == 0 {
		config.GA.LocalSearch.KDTreeNeighbors = 20
	}
	if config.GA.LocalSearch.KDTreeThreshold == 0 {
		config.GA.LocalSearch.KDTreeThreshold = 200
	}

	if config.GA.Island.NumIslands == 0 {
		config.GA.Island.NumIslands = 4
	}
	if config.GA.Island.Migration.Interval == 0 {
		config.GA.Island.Migration.Interval = 20
	}
	if config.GA.Island.Migration.NumMigrants == 0 {
		config.GA.Island.Migration.NumMigrants = 3
	}
	if config.GA.Island.Migration.Topology == "" {
		config.GA.Island.Migration.Topology = "ring"
	}

	if config.GA.NSGA2.NumObjectives == 0 {
		config.GA.NSGA2.NumObjectives = 2
	}

	if config.GA.TSP.DataFormat == "" {
		config.GA.TSP.DataFormat = "coordinates"
	}

	if config.GA.Output.CSVFile == "" {
		config.GA.Output.CSVFile = "convergence.csv"
	}
	if config.GA.Output.SVGFile == "" {
		config.GA.Output.SVGFile = "tsp_solution.svg"
	}
	if config.GA.Output.ResultFile == "" {
		config.GA.Output.ResultFile = "result.txt"
	}

	if config.Benchmark.Runs == 0 {
		config.Benchmark.Runs = 5
	}

	if config.TSPTW.PenaltyType == "" {
		config.TSPTW.PenaltyType = "fixed"
	}
	if config.TSPTW.PenaltyCoefficient == 0 {
		config.TSPTW.PenaltyCoefficient = 100.0
	}
	if config.TSPTW.FeasibilityTarget == 0 {
		config.TSPTW.FeasibilityTarget = 0.5
	}
	if config.TSPTW.PenaltyAdjustInterval == 0 {
		config.TSPTW.PenaltyAdjustInterval = 10
	}
	if config.TSPTW.RepairInterval == 0 {
		config.TSPTW.RepairInterval = 10
	}
	if config.TSPTW.RepairTopK == 0 {
		config.TSPTW.RepairTopK = 5
	}
	if config.TSPTW.Speed == 0 {
		config.TSPTW.Speed = 1.0
	}
	if config.TSPTW.PopulationSize == 0 {
		config.TSPTW.PopulationSize = 100
	}
	if config.TSPTW.Generations == 0 {
		config.TSPTW.Generations = 500
	}
	if config.TSPTW.CrossoverRate == 0 {
		config.TSPTW.CrossoverRate = 0.8
	}
	if config.TSPTW.MutationRate == 0 {
		config.TSPTW.MutationRate = 0.02
	}
	if config.TSPTW.TournamentSize == 0 {
		config.TSPTW.TournamentSize = 3
	}
	if config.TSPTW.EliteCount == 0 {
		config.TSPTW.EliteCount = 2
	}
}

func SaveConfig(config *Config, filePath string) error {
	data, err := yaml.Marshal(config)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %v", err)
	}

	err = os.WriteFile(filePath, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write config file: %v", err)
	}

	return nil
}
