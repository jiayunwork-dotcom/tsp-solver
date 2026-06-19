package tsp

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

type City struct {
	ID int
	X  float64
	Y  float64
}

type TSPProblem struct {
	Name          string
	NumCities     int
	Cities        []City
	DistanceMatrix [][]float64
	UseMatrix     bool
	OptimalTour   []int
	OptimalLength float64
}

type DataFormat string

const (
	Coordinates DataFormat = "coordinates"
	DistanceMatrix DataFormat = "distance_matrix"
	TSPLIBFormat DataFormat = "tsplib"
)

func LoadTSPProblem(filePath string, format DataFormat) (*TSPProblem, error) {
	switch format {
	case Coordinates:
		return loadCoordinates(filePath)
	case DistanceMatrix:
		return loadDistanceMatrix(filePath)
	case TSPLIBFormat:
		return LoadTSPLIB(filePath)
	default:
		return loadCoordinates(filePath)
	}
}

func loadCoordinates(filePath string) (*TSPProblem, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	var cities []City
	scanner := bufio.NewScanner(file)
	id := 0

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}

		x, err := strconv.ParseFloat(parts[0], 64)
		if err != nil {
			continue
		}
		y, err := strconv.ParseFloat(parts[1], 64)
		if err != nil {
			continue
		}

		cities = append(cities, City{ID: id, X: x, Y: y})
		id++
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %v", err)
	}

	if len(cities) == 0 {
		return nil, fmt.Errorf("no cities found in file")
	}

	problem := &TSPProblem{
		Name:      filePath,
		NumCities: len(cities),
		Cities:    cities,
		UseMatrix: false,
	}

	return problem, nil
}

func loadDistanceMatrix(filePath string) (*TSPProblem, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	var matrix [][]float64
	scanner := bufio.NewScanner(file)
	n := 0

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.Fields(line)
		row := make([]float64, len(parts))
		for i, p := range parts {
			val, err := strconv.ParseFloat(p, 64)
			if err != nil {
				return nil, fmt.Errorf("invalid distance value: %s", p)
			}
			row[i] = val
		}
		matrix = append(matrix, row)
		n++
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %v", err)
	}

	if n == 0 {
		return nil, fmt.Errorf("empty distance matrix")
	}

	cities := make([]City, n)
	for i := 0; i < n; i++ {
		cities[i] = City{ID: i, X: float64(i), Y: 0}
	}

	problem := &TSPProblem{
		Name:           filePath,
		NumCities:      n,
		Cities:         cities,
		DistanceMatrix: matrix,
		UseMatrix:      true,
	}

	return problem, nil
}

func (p *TSPProblem) Distance(i, j int) float64 {
	if p.UseMatrix {
		return p.DistanceMatrix[i][j]
	}
	return euclideanDistance(p.Cities[i], p.Cities[j])
}

func euclideanDistance(c1, c2 City) float64 {
	dx := c1.X - c2.X
	dy := c1.Y - c2.Y
	return math.Sqrt(dx*dx + dy*dy)
}

func (p *TSPProblem) TourLength(tour []int) float64 {
	if len(tour) < 2 {
		return 0
	}
	total := 0.0
	for i := 0; i < len(tour)-1; i++ {
		total += p.Distance(tour[i], tour[i+1])
	}
	total += p.Distance(tour[len(tour)-1], tour[0])
	return total
}

func (p *TSPProblem) Fitness(tour []int) float64 {
	length := p.TourLength(tour)
	if length < 1e-10 {
		return 1e10
	}
	return 1.0 / length
}

func (p *TSPProblem) ValidateTour(tour []int) bool {
	if len(tour) != p.NumCities {
		return false
	}
	visited := make(map[int]bool)
	for _, city := range tour {
		if city < 0 || city >= p.NumCities {
			return false
		}
		if visited[city] {
			return false
		}
		visited[city] = true
	}
	return len(visited) == p.NumCities
}
