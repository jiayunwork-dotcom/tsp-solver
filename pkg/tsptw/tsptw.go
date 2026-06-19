package tsptw

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/tsp-solver/pkg/utils"
)

type TWCity struct {
	ID          int
	X           float64
	Y           float64
	Earliest    float64
	Latest      float64
	ServiceTime float64
}

type TSPTWProblem struct {
	Name      string
	NumCities int
	Cities    []TWCity
	Speed     float64
}

type VisitInfo struct {
	CityID       int
	ArrivalTime  float64
	WaitTime     float64
	ServiceStart float64
	ServiceEnd   float64
	Violation    float64
}

type TourEvaluation struct {
	TotalDistance      float64
	TotalTravelTime    float64
	TotalWaitTime      float64
	TotalViolation     float64
	ReturnViolation    float64
	ReturnArrivalTime  float64
	Visits             []VisitInfo
	IsFeasible         bool
	PenalizedCost      float64
}

func LoadTSPTWProblem(filePath string, speed float64) (*TSPTWProblem, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	var cities []TWCity
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

		earliest := 0.0
		latest := 99999.0
		serviceTime := 0.0

		if len(parts) >= 5 {
			e, err := strconv.ParseFloat(parts[2], 64)
			if err == nil {
				earliest = e
			}
			l, err := strconv.ParseFloat(parts[3], 64)
			if err == nil {
				latest = l
			}
			s, err := strconv.ParseFloat(parts[4], 64)
			if err == nil {
				serviceTime = s
			}
		}

		cities = append(cities, TWCity{
			ID:          id,
			X:           x,
			Y:           y,
			Earliest:    earliest,
			Latest:      latest,
			ServiceTime: serviceTime,
		})
		id++
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %v", err)
	}

	if len(cities) == 0 {
		return nil, fmt.Errorf("no cities found in file")
	}

	return &TSPTWProblem{
		Name:      filePath,
		NumCities: len(cities),
		Cities:    cities,
		Speed:     speed,
	}, nil
}

func (p *TSPTWProblem) Distance(i, j int) float64 {
	dx := p.Cities[i].X - p.Cities[j].X
	dy := p.Cities[i].Y - p.Cities[j].Y
	return math.Sqrt(dx*dx + dy*dy)
}

func (p *TSPTWProblem) TravelTime(i, j int) float64 {
	return p.Distance(i, j) / p.Speed
}

func (p *TSPTWProblem) EvaluateTour(tour []int) *TourEvaluation {
	if len(tour) == 0 {
		return &TourEvaluation{}
	}

	n := p.NumCities
	orderedTour := normalizeTour(tour, n)

	eval := &TourEvaluation{
		Visits: make([]VisitInfo, len(orderedTour)),
	}

	totalDistance := 0.0
	currentTime := 0.0
	totalWait := 0.0
	totalViolation := 0.0

	for i, cityIdx := range orderedTour {
		var arrivalTime float64
		if i == 0 {
			arrivalTime = 0.0
		} else {
			prevCity := orderedTour[i-1]
			travelTime := p.TravelTime(prevCity, cityIdx)
			travelDist := p.Distance(prevCity, cityIdx)
			totalDistance += travelDist
			arrivalTime = currentTime + travelTime
		}

		city := p.Cities[cityIdx]
		rawArrival := arrivalTime
		var waitTime float64
		if arrivalTime < city.Earliest {
			waitTime = city.Earliest - arrivalTime
			arrivalTime = city.Earliest
		}

		totalWait += waitTime

		serviceStart := arrivalTime
		violation := 0.0
		if arrivalTime > city.Latest {
			violation = arrivalTime - city.Latest
		}
		totalViolation += violation

		serviceEnd := serviceStart + city.ServiceTime

		eval.Visits[i] = VisitInfo{
			CityID:       cityIdx,
			ArrivalTime:  rawArrival,
			WaitTime:     waitTime,
			ServiceStart: serviceStart,
			ServiceEnd:   serviceEnd,
			Violation:    violation,
		}

		currentTime = serviceEnd
	}

	lastCity := orderedTour[len(orderedTour)-1]
	firstCity := orderedTour[0]
	returnDist := p.Distance(lastCity, firstCity)
	returnTravelTime := p.TravelTime(lastCity, firstCity)
	totalDistance += returnDist

	depot := p.Cities[firstCity]
	returnArrivalTime := currentTime + returnTravelTime
	returnViolation := 0.0
	if returnArrivalTime > depot.Latest {
		returnViolation = returnArrivalTime - depot.Latest
	}
	totalViolation += returnViolation

	eval.TotalDistance = totalDistance
	eval.TotalTravelTime = totalDistance / p.Speed
	eval.TotalWaitTime = totalWait
	eval.TotalViolation = totalViolation
	eval.ReturnViolation = returnViolation
	eval.ReturnArrivalTime = returnArrivalTime
	eval.IsFeasible = totalViolation < 1e-10

	return eval
}

func (p *TSPTWProblem) ComputePenalizedFitness(tour []int, penaltyCoeff float64) float64 {
	eval := p.EvaluateTour(tour)
	distanceWithWait := eval.TotalDistance + eval.TotalWaitTime*p.Speed
	penalty := penaltyCoeff * eval.TotalViolation
	cost := distanceWithWait + penalty
	if cost < 1e-10 {
		return 1e10
	}
	return 1.0 / cost
}

func (p *TSPTWProblem) ComputePenalizedCost(tour []int, penaltyCoeff float64) float64 {
	eval := p.EvaluateTour(tour)
	distanceWithWait := eval.TotalDistance + eval.TotalWaitTime*p.Speed
	penalty := penaltyCoeff * eval.TotalViolation
	return distanceWithWait + penalty
}

type PenaltyController struct {
	PenaltyType           string
	PenaltyCoefficient    float64
	FeasibilityTarget     float64
	PenaltyAdjustInterval int
	CurrentCoefficient    float64
}

func NewPenaltyController(penaltyType string, coeff float64, target float64, interval int) *PenaltyController {
	return &PenaltyController{
		PenaltyType:           penaltyType,
		PenaltyCoefficient:    coeff,
		FeasibilityTarget:     target,
		PenaltyAdjustInterval: interval,
		CurrentCoefficient:    coeff,
	}
}

func (pc *PenaltyController) GetCoefficient() float64 {
	return pc.CurrentCoefficient
}

func (pc *PenaltyController) Adjust(feasibleRatio float64) {
	if pc.PenaltyType != "adaptive" {
		return
	}
	if feasibleRatio < pc.FeasibilityTarget {
		pc.CurrentCoefficient *= 1.2
	} else if feasibleRatio > pc.FeasibilityTarget+0.1 {
		pc.CurrentCoefficient *= 0.85
		if pc.CurrentCoefficient < 1.0 {
			pc.CurrentCoefficient = 1.0
		}
	}
}

func RepairIndividual(problem *TSPTWProblem, tour []int) []int {
	improved := true
	result := make([]int, len(tour))
	copy(result, tour)

	for improved {
		improved = false
		eval := problem.EvaluateTour(result)
		violations := 0.0
		for _, v := range eval.Visits {
			violations += v.Violation
		}
		if violations < 1e-10 {
			break
		}

		for i := 1; i < len(result)-1; i++ {
			newTour := make([]int, len(result))
			copy(newTour, result)
			newTour[i], newTour[i+1] = newTour[i+1], newTour[i]

			if i > 0 {
				prevCity := newTour[i-1]
				cityA := newTour[i]
				cityB := newTour[i+1]
				travelBefore := problem.TravelTime(prevCity, cityA)
				travelOld := problem.TravelTime(prevCity, cityB)
				if travelBefore > travelOld {
					continue
				}
			}

			newEval := problem.EvaluateTour(newTour)
			newViolations := 0.0
			for _, v := range newEval.Visits {
				newViolations += v.Violation
			}

			if newViolations < violations-1e-10 {
				result = newTour
				improved = true
				break
			}
		}
	}

	return result
}

type individualInfo struct {
	Index      int
	Violations float64
}

func RepairTopK(problem *TSPTWProblem, population [][]int, topK int) {
	violated := make([]individualInfo, 0)
	for i, tour := range population {
		eval := problem.EvaluateTour(tour)
		if eval.TotalViolation > 1e-10 {
			violated = append(violated, individualInfo{Index: i, Violations: eval.TotalViolation})
		}
	}

	sort.Slice(violated, func(i, j int) bool {
		return violated[i].Violations > violated[j].Violations
	})

	if topK > len(violated) {
		topK = len(violated)
	}

	for i := 0; i < topK; i++ {
		idx := violated[i].Index
		repaired := RepairIndividual(problem, population[idx])
		population[idx] = repaired
	}
}

func ComputeFeasibleRatio(problem *TSPTWProblem, population [][]int) float64 {
	if len(population) == 0 {
		return 0
	}
	feasible := 0
	for _, tour := range population {
		eval := problem.EvaluateTour(tour)
		if eval.IsFeasible {
			feasible++
		}
	}
	return float64(feasible) / float64(len(population))
}

func tournamentSelect(population [][]int, fitnesses []float64, tournamentSize int) int {
	bestIdx := utils.RandInt(0, len(population)-1)
	for i := 1; i < tournamentSize; i++ {
		candidate := utils.RandInt(0, len(population)-1)
		if fitnesses[candidate] > fitnesses[bestIdx] {
			bestIdx = candidate
		}
	}
	return bestIdx
}

func oxCrossover(parent1, parent2 []int) ([]int, []int) {
	n := len(parent1)
	if n <= 2 {
		return copyTour(parent1), copyTour(parent2)
	}

	child1 := make([]int, n)
	child2 := make([]int, n)
	child1[0] = 0
	child2[0] = 0

	m := n - 1
	if m <= 2 {
		copy(child1[1:], parent1[1:])
		copy(child2[1:], parent2[1:])
		return child1, child2
	}

	start := utils.RandInt(0, m-2) + 1
	end := utils.RandInt(start, n-1)

	p1Sub := parent1[start : end+1]
	p2Sub := parent2[start : end+1]

	for i := range child1 {
		child1[i] = -1
		child2[i] = -1
	}
	child1[0] = 0
	child2[0] = 0

	copy(child1[start:end+1], p1Sub)
	copy(child2[start:end+1], p2Sub)

	fillOXFrom(child1, parent2, start, end, n)
	fillOXFrom(child2, parent1, start, end, n)

	return child1, child2
}

func fillOXFrom(child []int, parent []int, start, end, n int) {
	used := make(map[int]bool)
	for _, v := range child {
		if v != -1 {
			used[v] = true
		}
	}

	pos := 1
	parentPos := 0
	for pos < n {
		if child[pos] != -1 {
			pos++
			continue
		}
		if parentPos >= n {
			parentPos = 0
		}
		val := parent[parentPos]
		parentPos++
		if val == 0 || used[val] {
			continue
		}
		child[pos] = val
		used[val] = true
		pos++
	}
}

func swapMutate(tour []int) []int {
	result := copyTour(tour)
	n := len(result)
	if n < 3 {
		return result
	}
	i := utils.RandInt(1, n-1)
	j := utils.RandInt(1, n-1)
	for j == i {
		j = utils.RandInt(1, n-1)
	}
	result[i], result[j] = result[j], result[i]
	return result
}

func insertMutate(tour []int) []int {
	result := copyTour(tour)
	n := len(result)
	if n < 4 {
		return result
	}
	i := utils.RandInt(1, n-1)
	j := utils.RandInt(1, n-1)
	for j == i {
		j = utils.RandInt(1, n-1)
	}

	val := result[i]
	if i < j {
		copy(result[i:j], result[i+1:j+1])
		result[j] = val
	} else {
		copy(result[j+1:i+1], result[j:i])
		result[j] = val
	}
	return result
}

func copyTour(tour []int) []int {
	t := make([]int, len(tour))
	copy(t, tour)
	return t
}

func RandomTour(n int) []int {
	return randomTour(n)
}

func randomTour(n int) []int {
	if n <= 1 {
		tour := make([]int, n)
		for i := range tour {
			tour[i] = i
		}
		return tour
	}
	tour := make([]int, n)
	tour[0] = 0
	rest := make([]int, n-1)
	for i := 1; i < n; i++ {
		rest[i-1] = i
	}
	utils.Shuffle(rest)
	copy(tour[1:], rest)
	return tour
}

func ensureDepotFirst(tour []int) []int {
	if len(tour) == 0 {
		return tour
	}
	if tour[0] == 0 {
		return tour
	}
	result := make([]int, len(tour))
	depotIdx := -1
	for i, v := range tour {
		if v == 0 {
			depotIdx = i
			break
		}
	}
	if depotIdx == -1 {
		result[0] = 0
		idx := 1
		for _, v := range tour {
			if v != 0 {
				result[idx] = v
				idx++
			}
		}
		return result
	}
	copy(result, tour[depotIdx:])
	copy(result[len(tour)-depotIdx:], tour[:depotIdx])
	return result
}

func normalizeTour(tour []int, n int) []int {
	result := make([]int, n)
	for i := range result {
		result[i] = -1
	}
	result[0] = 0

	used := make(map[int]bool)
	used[0] = true

	pos := 1
	for _, v := range tour {
		if v < 0 || v >= n {
			continue
		}
		if used[v] {
			continue
		}
		if pos >= n {
			break
		}
		result[pos] = v
		used[v] = true
		pos++
	}

	for v := 1; v < n && pos < n; v++ {
		if !used[v] {
			result[pos] = v
			pos++
		}
	}

	return result
}
