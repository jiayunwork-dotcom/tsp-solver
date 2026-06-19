package tsp

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type TSPLIBSpec struct {
	Name        string
	Type        string
	Comment     string
	Dimension   int
	EdgeWeightType string
	EdgeWeightFormat string
	DisplayDataType string
}

func LoadTSPLIB(filePath string) (*TSPProblem, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open TSPLIB file: %v", err)
	}
	defer file.Close()

	spec := TSPLIBSpec{}
	var cities []City
	var distanceMatrix [][]float64
	nodeCoordSection := false
	edgeWeightSection := false
	displayDataSection := false

	scanner := bufio.NewScanner(file)
	lineNum := 0

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		lineNum++

		if line == "" {
			continue
		}

		if strings.HasPrefix(line, "NAME") {
			spec.Name = strings.TrimSpace(strings.TrimPrefix(line, "NAME:"))
			continue
		}
		if strings.HasPrefix(line, "TYPE") {
			spec.Type = strings.TrimSpace(strings.TrimPrefix(line, "TYPE:"))
			continue
		}
		if strings.HasPrefix(line, "COMMENT") {
			spec.Comment = strings.TrimSpace(strings.TrimPrefix(line, "COMMENT:"))
			continue
		}
		if strings.HasPrefix(line, "DIMENSION") {
			dimStr := strings.TrimSpace(strings.TrimPrefix(line, "DIMENSION:"))
			spec.Dimension, err = strconv.Atoi(dimStr)
			if err != nil {
				return nil, fmt.Errorf("invalid dimension at line %d", lineNum)
			}
			continue
		}
		if strings.HasPrefix(line, "EDGE_WEIGHT_TYPE") {
			spec.EdgeWeightType = strings.TrimSpace(strings.TrimPrefix(line, "EDGE_WEIGHT_TYPE:"))
			continue
		}
		if strings.HasPrefix(line, "EDGE_WEIGHT_FORMAT") {
			spec.EdgeWeightFormat = strings.TrimSpace(strings.TrimPrefix(line, "EDGE_WEIGHT_FORMAT:"))
			continue
		}
		if strings.HasPrefix(line, "DISPLAY_DATA_TYPE") {
			spec.DisplayDataType = strings.TrimSpace(strings.TrimPrefix(line, "DISPLAY_DATA_TYPE:"))
			continue
		}

		if line == "NODE_COORD_SECTION" {
			nodeCoordSection = true
			cities = make([]City, spec.Dimension)
			continue
		}
		if line == "EDGE_WEIGHT_SECTION" {
			edgeWeightSection = true
			distanceMatrix = make([][]float64, spec.Dimension)
			for i := range distanceMatrix {
				distanceMatrix[i] = make([]float64, spec.Dimension)
			}
			continue
		}
		if line == "DISPLAY_DATA_SECTION" {
			displayDataSection = true
			continue
		}
		if line == "EOF" {
			break
		}

		if nodeCoordSection {
			parts := strings.Fields(line)
			if len(parts) < 3 {
				continue
			}
			id, _ := strconv.Atoi(parts[0])
			x, _ := strconv.ParseFloat(parts[1], 64)
			y, _ := strconv.ParseFloat(parts[2], 64)
			if id-1 < len(cities) {
				cities[id-1] = City{ID: id - 1, X: x, Y: y}
			}
		}

		if edgeWeightSection {
			parts := strings.Fields(line)
			if spec.EdgeWeightFormat == "FULL_MATRIX" {
				rowIdx := 0
				for i, v := range parts {
					val, _ := strconv.ParseFloat(v, 64)
					if rowIdx < spec.Dimension && i < spec.Dimension {
						distanceMatrix[rowIdx][i] = val
					}
				}
				rowIdx++
			} else if spec.EdgeWeightFormat == "LOWER_DIAG_ROW" {
			}
		}

		if displayDataSection {
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading TSPLIB file: %v", err)
	}

	problem := &TSPProblem{
		Name:      spec.Name,
		NumCities: spec.Dimension,
	}

	if len(cities) > 0 {
		problem.Cities = cities
		problem.UseMatrix = false
	} else if distanceMatrix != nil {
		problem.DistanceMatrix = distanceMatrix
		problem.UseMatrix = true
		problem.Cities = make([]City, spec.Dimension)
		for i := 0; i < spec.Dimension; i++ {
			problem.Cities[i] = City{ID: i, X: float64(i), Y: 0}
		}
	} else {
		return nil, fmt.Errorf("no city coordinates or distance matrix found")
	}

	return problem, nil
}

func LoadOptimalTour(filePath string) ([]int, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var tour []int
	tourSection := false

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "TOUR_SECTION" {
			tourSection = true
			continue
		}
		if line == "-1" || line == "EOF" {
			break
		}
		if tourSection {
			id, err := strconv.Atoi(line)
			if err == nil && id > 0 {
				tour = append(tour, id-1)
			}
		}
	}

	return tour, scanner.Err()
}
