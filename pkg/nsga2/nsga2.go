package nsga2

import (
	"sort"
	"time"

	"github.com/tsp-solver/pkg/ga/crossover"
	"github.com/tsp-solver/pkg/ga/encoding"
	"github.com/tsp-solver/pkg/ga/mutation"
	"github.com/tsp-solver/pkg/utils"
)

type MultiObjectiveFitnessFunc func(ind *encoding.Individual) []float64

type NSGA2Config struct {
	EncodingType    encoding.EncodingType
	GenomeLength    int
	PermutationSize int
	Bounds          [][2]float64
	PopulationSize  int
	Generations     int
	NumObjectives   int

	CrossoverConfig crossover.Config
	MutationConfig  mutation.Config

	FitnessFunction MultiObjectiveFitnessFunc
}

type NSGA2Result struct {
	ParetoFront []*encoding.Individual
	History     *NSGA2History
	Duration    time.Duration
}

type NSGA2History struct {
	Generations []int
	HV          []float64
	NumFronts   []int
}

type IndividualMO struct {
	*encoding.Individual
	Objectives    []float64
	Rank          int
	CrowdingDist  float64
	DominatedCount int
	Dominates     []int
}

type NSGA2 struct {
	Config  NSGA2Config
	Pop     []*IndividualMO
	Gen     int
	History *NSGA2History
}

func NewNSGA2(config NSGA2Config) *NSGA2 {
	nsga := &NSGA2{
		Config: config,
		History: &NSGA2History{
			Generations: make([]int, 0),
			HV:          make([]float64, 0),
			NumFronts:   make([]int, 0),
		},
	}
	nsga.initPopulation()
	return nsga
}

func (nsga *NSGA2) initPopulation() {
	pop := encoding.NewPopulation(
		nsga.Config.PopulationSize,
		nsga.Config.GenomeLength,
		nsga.Config.EncodingType,
		nsga.Config.PermutationSize,
		nsga.Config.Bounds,
	)

	nsga.Pop = make([]*IndividualMO, len(pop))
	for i, ind := range pop {
		moInd := &IndividualMO{
			Individual: ind,
			Objectives: make([]float64, nsga.Config.NumObjectives),
		}
		nsga.evaluate(moInd)
		nsga.Pop[i] = moInd
	}

	nsga.fastNonDominatedSort(nsga.Pop)
	nsga.crowdingDistanceAssignment(nsga.Pop)
}

func (nsga *NSGA2) evaluate(ind *IndividualMO) {
	if !ind.Evaluated && nsga.Config.FitnessFunction != nil {
		ind.Objectives = nsga.Config.FitnessFunction(ind.Individual)
		ind.Evaluated = true
	}
}

func (nsga *NSGA2) dominates(p, q *IndividualMO) bool {
	better := false
	worse := false

	for i := 0; i < nsga.Config.NumObjectives; i++ {
		if p.Objectives[i] < q.Objectives[i] {
			better = true
		} else if p.Objectives[i] > q.Objectives[i] {
			worse = true
		}
	}

	return better && !worse
}

func (nsga *NSGA2) fastNonDominatedSort(pop []*IndividualMO) [][]*IndividualMO {
	n := len(pop)

	for i := 0; i < n; i++ {
		pop[i].DominatedCount = 0
		pop[i].Dominates = make([]int, 0)
	}

	var fronts [][]*IndividualMO
	front1 := make([]*IndividualMO, 0)

	for p := 0; p < n; p++ {
		for q := 0; q < n; q++ {
			if p == q {
				continue
			}
			if nsga.dominates(pop[p], pop[q]) {
				pop[p].Dominates = append(pop[p].Dominates, q)
			} else if nsga.dominates(pop[q], pop[p]) {
				pop[p].DominatedCount++
			}
		}
		if pop[p].DominatedCount == 0 {
			pop[p].Rank = 0
			front1 = append(front1, pop[p])
		}
	}

	fronts = append(fronts, front1)
	i := 0

	for len(fronts[i]) > 0 {
		nextFront := make([]*IndividualMO, 0)
		for _, p := range fronts[i] {
			for _, qIdx := range p.Dominates {
				pop[qIdx].DominatedCount--
				if pop[qIdx].DominatedCount == 0 {
					pop[qIdx].Rank = i + 1
					nextFront = append(nextFront, pop[qIdx])
				}
			}
		}
		i++
		if len(nextFront) > 0 {
			fronts = append(fronts, nextFront)
		} else {
			break
		}
	}

	return fronts
}

func (nsga *NSGA2) crowdingDistanceAssignment(front []*IndividualMO) {
	n := len(front)
	if n == 0 {
		return
	}

	for i := range front {
		front[i].CrowdingDist = 0
	}

	for obj := 0; obj < nsga.Config.NumObjectives; obj++ {
		sorted := make([]*IndividualMO, n)
		copy(sorted, front)
		sort.Slice(sorted, func(i, j int) bool {
			return sorted[i].Objectives[obj] < sorted[j].Objectives[obj]
		})

		sorted[0].CrowdingDist = 1e10
		sorted[n-1].CrowdingDist = 1e10

		if n <= 2 {
			continue
		}

		min := sorted[0].Objectives[obj]
		max := sorted[n-1].Objectives[obj]
		range_ := max - min

		if range_ < 1e-10 {
			continue
		}

		for i := 1; i < n-1; i++ {
			dist := (sorted[i+1].Objectives[obj] - sorted[i-1].Objectives[obj]) / range_
			sorted[i].CrowdingDist += dist
		}
	}
}

func (nsga *NSGA2) crowdingComp(a, b *IndividualMO) bool {
	if a.Rank < b.Rank {
		return true
	}
	if a.Rank == b.Rank && a.CrowdingDist > b.CrowdingDist {
		return true
	}
	return false
}

func (nsga *NSGA2) tournamentSelect(pop []*IndividualMO, k int) *IndividualMO {
	bestIdx := utils.RandInt(0, len(pop)-1)
	for i := 1; i < k; i++ {
		idx := utils.RandInt(0, len(pop)-1)
		if nsga.crowdingComp(pop[idx], pop[bestIdx]) {
			bestIdx = idx
		}
	}
	return pop[bestIdx]
}

func (nsga *NSGA2) Step() bool {
	if nsga.Gen >= nsga.Config.Generations {
		return false
	}

	offspringMO := make([]*IndividualMO, 0, nsga.Config.PopulationSize)

	for len(offspringMO) < nsga.Config.PopulationSize {
		parent1 := nsga.tournamentSelect(nsga.Pop, 2)
		parent2 := nsga.tournamentSelect(nsga.Pop, 2)

		child1, child2 := crossover.Crossover(
			parent1.Individual,
			parent2.Individual,
			nsga.Config.EncodingType,
			nsga.Config.CrossoverConfig,
		)

		child1 = mutation.Mutate(child1, nsga.Config.EncodingType, nsga.Config.MutationConfig)
		child2 = mutation.Mutate(child2, nsga.Config.EncodingType, nsga.Config.MutationConfig)

		mo1 := &IndividualMO{
			Individual: child1,
			Objectives: make([]float64, nsga.Config.NumObjectives),
		}
		mo2 := &IndividualMO{
			Individual: child2,
			Objectives: make([]float64, nsga.Config.NumObjectives),
		}

		nsga.evaluate(mo1)
		nsga.evaluate(mo2)

		offspringMO = append(offspringMO, mo1, mo2)
	}

	offspringMO = offspringMO[:nsga.Config.PopulationSize]

	combined := make([]*IndividualMO, 0, len(nsga.Pop)+len(offspringMO))
	combined = append(combined, nsga.Pop...)
	combined = append(combined, offspringMO...)

	fronts := nsga.fastNonDominatedSort(combined)

	newPop := make([]*IndividualMO, 0, nsga.Config.PopulationSize)
	frontIdx := 0

	for frontIdx < len(fronts) && len(newPop)+len(fronts[frontIdx]) <= nsga.Config.PopulationSize {
		nsga.crowdingDistanceAssignment(fronts[frontIdx])
		newPop = append(newPop, fronts[frontIdx]...)
		frontIdx++
	}

	if frontIdx < len(fronts) && len(newPop) < nsga.Config.PopulationSize {
		front := fronts[frontIdx]
		nsga.crowdingDistanceAssignment(front)

		sortedFront := make([]*IndividualMO, len(front))
		copy(sortedFront, front)
		sort.Slice(sortedFront, func(i, j int) bool {
			return nsga.crowdingComp(sortedFront[i], sortedFront[j])
		})

		remaining := nsga.Config.PopulationSize - len(newPop)
		newPop = append(newPop, sortedFront[:remaining]...)
	}

	nsga.Pop = newPop
	nsga.Gen++

	fronts = nsga.fastNonDominatedSort(nsga.Pop)
	nsga.History.Generations = append(nsga.History.Generations, nsga.Gen)
	nsga.History.NumFronts = append(nsga.History.NumFronts, len(fronts))

	hv := nsga.calculateHV()
	nsga.History.HV = append(nsga.History.HV, hv)

	return true
}

func (nsga *NSGA2) calculateHV() float64 {
	if len(nsga.Pop) == 0 {
		return 0
	}

	fronts := nsga.fastNonDominatedSort(nsga.Pop)
	if len(fronts) == 0 {
		return 0
	}

	paretoFront := fronts[0]
	if len(paretoFront) == 0 {
		return 0
	}

	numObj := nsga.Config.NumObjectives
	if numObj <= 0 {
		return 0
	}

	refPoint := make([]float64, numObj)
	for i := range refPoint {
		maxVal := paretoFront[0].Objectives[i]
		for _, ind := range paretoFront {
			if ind.Objectives[i] > maxVal {
				maxVal = ind.Objectives[i]
			}
		}
		minVal := paretoFront[0].Objectives[i]
		for _, ind := range paretoFront {
			if ind.Objectives[i] < minVal {
				minVal = ind.Objectives[i]
			}
		}
		range_ := maxVal - minVal
		if range_ < 1e-10 {
			range_ = 1.0
		}
		refPoint[i] = maxVal + range_*0.1
	}

	points := make([][]float64, len(paretoFront))
	for i, ind := range paretoFront {
		points[i] = make([]float64, numObj)
		copy(points[i], ind.Objectives)
	}

	if numObj == 2 {
		return calculateHV2D(points, refPoint)
	}

	return calculateHVMonteCarlo(points, refPoint, 100000)
}

func calculateHV2D(points [][]float64, refPoint []float64) float64 {
	if len(points) == 0 {
		return 0
	}

	normalized := make([][]float64, len(points))
	for i, p := range points {
		normalized[i] = []float64{
			refPoint[0] - p[0],
			refPoint[1] - p[1],
		}
		if normalized[i][0] < 0 {
			normalized[i][0] = 0
		}
		if normalized[i][1] < 0 {
			normalized[i][1] = 0
		}
	}

	sort.Slice(normalized, func(i, j int) bool {
		return normalized[i][0] < normalized[j][0]
	})

	hv := 0.0
	prevY := 0.0
	for i := 0; i < len(normalized); i++ {
		width := normalized[i][0]
		if i > 0 {
			width = normalized[i][0] - normalized[i-1][0]
		}
		height := normalized[i][1]
		if height > prevY {
			prevY = height
		}
		hv += width * prevY
	}

	return hv
}

func calculateHVMonteCarlo(points [][]float64, refPoint []float64, samples int) float64 {
	if len(points) == 0 || samples <= 0 {
		return 0
	}

	numObj := len(refPoint)

	minVals := make([]float64, numObj)
	for i := range minVals {
		minVals[i] = points[0][i]
		for _, p := range points {
			if p[i] < minVals[i] {
				minVals[i] = p[i]
			}
		}
	}

	totalVolume := 1.0
	rangeVals := make([]float64, numObj)
	for i := range refPoint {
		rangeVals[i] = refPoint[i] - minVals[i]
		if rangeVals[i] <= 0 {
			rangeVals[i] = 1.0
		}
		totalVolume *= rangeVals[i]
	}

	count := 0
	randSource := utils.GetRand()

	for s := 0; s < samples; s++ {
		sample := make([]float64, numObj)
		for i := range sample {
			sample[i] = minVals[i] + randSource.Float64()*rangeVals[i]
		}

		dominated := false
		for _, p := range points {
			dominatesAll := true
			for i := range p {
				if p[i] > sample[i] {
					dominatesAll = false
					break
				}
			}
			if dominatesAll {
				dominated = true
				break
			}
		}

		if dominated {
			count++
		}
	}

	return totalVolume * float64(count) / float64(samples)
}

func (nsga *NSGA2) Run() *NSGA2Result {
	start := time.Now()

	for nsga.Step() {
		if nsga.Gen%10 == 0 {
			fronts := nsga.fastNonDominatedSort(nsga.Pop)
			println("Generation", nsga.Gen, "Pareto front size:", len(fronts[0]))
		}
	}

	fronts := nsga.fastNonDominatedSort(nsga.Pop)
	paretoFront := make([]*encoding.Individual, 0)
	if len(fronts) > 0 {
		for _, ind := range fronts[0] {
			paretoFront = append(paretoFront, ind.Individual)
		}
	}

	return &NSGA2Result{
		ParetoFront: paretoFront,
		History:     nsga.History,
		Duration:    time.Since(start),
	}
}

func (nsga *NSGA2) GetParetoFront() []*encoding.Individual {
	fronts := nsga.fastNonDominatedSort(nsga.Pop)
	if len(fronts) == 0 {
		return nil
	}
	result := make([]*encoding.Individual, len(fronts[0]))
	for i, ind := range fronts[0] {
		result[i] = ind.Individual
	}
	return result
}
