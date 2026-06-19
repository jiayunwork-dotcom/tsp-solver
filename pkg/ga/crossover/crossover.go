package crossover

import (
	"math"

	"github.com/tsp-solver/pkg/ga/encoding"
	"github.com/tsp-solver/pkg/utils"
)

type CrossoverType string

const (
	PMX CrossoverType = "pmx"
	OX  CrossoverType = "ox"
	CX  CrossoverType = "cx"

	SinglePoint CrossoverType = "single_point"
	TwoPoint    CrossoverType = "two_point"
	Uniform     CrossoverType = "uniform"

	SBX CrossoverType = "sbx"
)

type Config struct {
	Type     CrossoverType
	Rate     float64
	SBX_eta  float64
	Bounds   [][2]float64
}

func Crossover(parent1, parent2 *encoding.Individual, encType encoding.EncodingType, config Config) (*encoding.Individual, *encoding.Individual) {
	if utils.RandFloat() > config.Rate {
		return parent1.Copy(), parent2.Copy()
	}

	switch encType {
	case encoding.PermutationEncoding:
		return permutationCrossover(parent1, parent2, config.Type)
	case encoding.BinaryEncoding:
		return binaryCrossover(parent1, parent2, config.Type)
	case encoding.RealEncoding:
		return realCrossover(parent1, parent2, config.Type, config)
	default:
		return parent1.Copy(), parent2.Copy()
	}
}

func permutationCrossover(p1, p2 *encoding.Individual, ct CrossoverType) (*encoding.Individual, *encoding.Individual) {
	perm1 := p1.GetPermutation()
	perm2 := p2.GetPermutation()
	var child1, child2 []int

	switch ct {
	case PMX:
		child1, child2 = pmxCrossover(perm1, perm2)
	case OX:
		child1, child2 = oxCrossover(perm1, perm2)
	case CX:
		child1, child2 = cxCrossover(perm1, perm2)
	default:
		child1, child2 = pmxCrossover(perm1, perm2)
	}

	c1 := p1.Copy()
	c2 := p2.Copy()
	c1.SetPermutation(child1)
	c2.SetPermutation(child2)
	c1.Evaluated = false
	c2.Evaluated = false
	return c1, c2
}

func pmxCrossover(p1, p2 []int) ([]int, []int) {
	n := len(p1)
	c1 := make([]int, n)
	c2 := make([]int, n)
	copy(c1, p1)
	copy(c2, p2)

	start := utils.RandInt(0, n-2)
	end := utils.RandInt(start+1, n-1)

	mapping1 := make(map[int]int)
	mapping2 := make(map[int]int)

	for i := start; i <= end; i++ {
		mapping1[p1[i]] = p2[i]
		mapping2[p2[i]] = p1[i]
		c1[i] = p2[i]
		c2[i] = p1[i]
	}

	for i := 0; i < n; i++ {
		if i >= start && i <= end {
			continue
		}
		val := p1[i]
		for containsKey(mapping1, val) {
			val = mapping1[val]
		}
		c1[i] = val

		val = p2[i]
		for containsKey(mapping2, val) {
			val = mapping2[val]
		}
		c2[i] = val
	}

	return c1, c2
}

func containsKey(m map[int]int, key int) bool {
	_, ok := m[key]
	return ok
}

func oxCrossover(p1, p2 []int) ([]int, []int) {
	n := len(p1)
	start := utils.RandInt(0, n-2)
	end := utils.RandInt(start+1, n-1)

	c1 := make([]int, n)
	c2 := make([]int, n)
	for i := range c1 {
		c1[i] = -1
		c2[i] = -1
	}

	for i := start; i <= end; i++ {
		c1[i] = p1[i]
		c2[i] = p2[i]
	}

	fillOX(c1, p2, start, end)
	fillOX(c2, p1, start, end)

	return c1, c2
}

func fillOX(child, parent []int, start, end int) {
	n := len(child)
	pos := (end + 1) % n
	for i := 0; i < n; i++ {
		idx := (end + 1 + i) % n
		val := parent[idx]
		if !containsVal(child, val) {
			child[pos] = val
			pos = (pos + 1) % n
		}
	}
}

func containsVal(arr []int, val int) bool {
	for _, v := range arr {
		if v == val {
			return true
		}
	}
	return false
}

func cxCrossover(p1, p2 []int) ([]int, []int) {
	n := len(p1)
	c1 := make([]int, n)
	c2 := make([]int, n)
	copy(c1, p2)
	copy(c2, p1)

	posMap1 := make(map[int]int)
	posMap2 := make(map[int]int)
	for i, v := range p1 {
		posMap1[v] = i
	}
	for i, v := range p2 {
		posMap2[v] = i
	}

	visited := make([]bool, n)
	useCycle := true

	for i := 0; i < n; i++ {
		if visited[i] {
			continue
		}

		cycle := []int{}
		cur := i
		for !visited[cur] {
			visited[cur] = true
			cycle = append(cycle, cur)
			val := p2[cur]
			cur = posMap1[val]
		}

		if useCycle {
			for _, idx := range cycle {
				c1[idx] = p1[idx]
				c2[idx] = p2[idx]
			}
		}
		useCycle = !useCycle
	}

	return c1, c2
}

func binaryCrossover(p1, p2 *encoding.Individual, ct CrossoverType) (*encoding.Individual, *encoding.Individual) {
	c1 := p1.Copy()
	c2 := p2.Copy()
	n := len(p1.Genome)

	switch ct {
	case SinglePoint:
		point := utils.RandInt(1, n-1)
		for i := point; i < n; i++ {
			c1.Genome[i], c2.Genome[i] = c2.Genome[i], c1.Genome[i]
		}
	case TwoPoint:
		p1 := utils.RandInt(1, n-2)
		p2 := utils.RandInt(p1+1, n-1)
		for i := p1; i < p2; i++ {
			c1.Genome[i], c2.Genome[i] = c2.Genome[i], c1.Genome[i]
		}
	case Uniform:
		for i := 0; i < n; i++ {
			if utils.RandFloat() < 0.5 {
				c1.Genome[i], c2.Genome[i] = c2.Genome[i], c1.Genome[i]
			}
		}
	}

	c1.Evaluated = false
	c2.Evaluated = false
	return c1, c2
}

func realCrossover(p1, p2 *encoding.Individual, ct CrossoverType, config Config) (*encoding.Individual, *encoding.Individual) {
	if ct == SBX {
		return sbxCrossover(p1, p2, config)
	}

	c1 := p1.Copy()
	c2 := p2.Copy()
	n := len(p1.Genome)

	switch ct {
	case SinglePoint:
		point := utils.RandInt(1, n-1)
		for i := point; i < n; i++ {
			c1.Genome[i], c2.Genome[i] = c2.Genome[i], c1.Genome[i]
		}
	case TwoPoint:
		p1Idx := utils.RandInt(1, n-2)
		p2Idx := utils.RandInt(p1Idx+1, n-1)
		for i := p1Idx; i < p2Idx; i++ {
			c1.Genome[i], c2.Genome[i] = c2.Genome[i], c1.Genome[i]
		}
	case Uniform:
		for i := 0; i < n; i++ {
			if utils.RandFloat() < 0.5 {
				c1.Genome[i], c2.Genome[i] = c2.Genome[i], c1.Genome[i]
			}
		}
	}

	c1.Evaluated = false
	c2.Evaluated = false
	return c1, c2
}

func sbxCrossover(p1, p2 *encoding.Individual, config Config) (*encoding.Individual, *encoding.Individual) {
	c1 := p1.Copy()
	c2 := p2.Copy()
	eta := config.SBX_eta
	if eta <= 0 {
		eta = 20
	}

	for i := range p1.Genome {
		if utils.RandFloat() > 0.5 {
			continue
		}

		y1 := p1.Genome[i]
		y2 := p2.Genome[i]

		if y1 > y2 {
			y1, y2 = y2, y1
		}

		low, high := 0.0, 1.0
		if i < len(config.Bounds) {
			low = config.Bounds[i][0]
			high = config.Bounds[i][1]
		}

		if y2-y1 > 1e-10 {
			rand := utils.RandFloat()

			beta1 := 1.0 + (2.0*(y1-low)/(y2-y1))
			beta2 := 1.0 + (2.0*(high-y2)/(y2-y1))
			alpha1 := 2.0 - math.Pow(beta1, -(eta+1))
			alpha2 := 2.0 - math.Pow(beta2, -(eta+1))

			var betaq1, betaq2 float64
			if rand <= 1.0/alpha1 {
				betaq1 = math.Pow(rand*alpha1, 1.0/(eta+1))
			} else {
				betaq1 = math.Pow(1.0/(2.0-rand*alpha1), 1.0/(eta+1))
			}

			rand = utils.RandFloat()
			if rand <= 1.0/alpha2 {
				betaq2 = math.Pow(rand*alpha2, 1.0/(eta+1))
			} else {
				betaq2 = math.Pow(1.0/(2.0-rand*alpha2), 1.0/(eta+1))
			}

			c1.Genome[i] = 0.5 * ((y1 + y2) - betaq1*(y2-y1))
			c2.Genome[i] = 0.5 * ((y1 + y2) + betaq2*(y2-y1))
		}

		if i < len(config.Bounds) {
			low := config.Bounds[i][0]
			high := config.Bounds[i][1]
			if c1.Genome[i] < low {
				c1.Genome[i] = low
			}
			if c1.Genome[i] > high {
				c1.Genome[i] = high
			}
			if c2.Genome[i] < low {
				c2.Genome[i] = low
			}
			if c2.Genome[i] > high {
				c2.Genome[i] = high
			}
		}
	}

	c1.Evaluated = false
	c2.Evaluated = false
	return c1, c2
}
