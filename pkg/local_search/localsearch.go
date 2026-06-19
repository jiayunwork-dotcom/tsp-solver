package local_search

import (
	"math"
	"sort"
)

type Point struct {
	X  float64
	Y  float64
	ID int
}

type DistanceFunc func(i, j int) float64

type KDTreeNode struct {
	Point Point
	Left  *KDTreeNode
	Right *KDTreeNode
	Axis  int
}

type KDTree struct {
	Root *KDTreeNode
	N    int
}

func BuildKDTree(points []Point) *KDTree {
	tree := &KDTree{N: len(points)}
	pts := make([]Point, len(points))
	copy(pts, points)
	tree.Root = buildKDTreeRecursive(pts, 0)
	return tree
}

func buildKDTreeRecursive(points []Point, depth int) *KDTreeNode {
	if len(points) == 0 {
		return nil
	}

	axis := depth % 2
	sorted := make([]Point, len(points))
	copy(sorted, points)
	sort.Slice(sorted, func(i, j int) bool {
		if axis == 0 {
			return sorted[i].X < sorted[j].X
		}
		return sorted[i].Y < sorted[j].Y
	})

	median := len(sorted) / 2
	node := &KDTreeNode{
		Point: sorted[median],
		Axis:  axis,
	}

	node.Left = buildKDTreeRecursive(sorted[:median], depth+1)
	node.Right = buildKDTreeRecursive(sorted[median+1:], depth+1)

	return node
}

func (tree *KDTree) NearestNeighbors(target Point, k int) []Point {
	if tree.Root == nil || k <= 0 {
		return nil
	}

	neighbors := make([]Point, 0, k)
	distances := make([]float64, 0, k)

	tree.nearestNeighborsRecursive(tree.Root, target, k, &neighbors, &distances)

	return neighbors
}

func (tree *KDTree) nearestNeighborsRecursive(node *KDTreeNode, target Point, k int, neighbors *[]Point, distances *[]float64) {
	if node == nil {
		return
	}

	dist := euclideanDistance(node.Point, target)

	if len(*neighbors) < k {
		*neighbors = append(*neighbors, node.Point)
		*distances = append(*distances, dist)
		insertSorted(neighbors, distances)
	} else if dist < (*distances)[len(*distances)-1] {
		(*neighbors)[len(*neighbors)-1] = node.Point
		(*distances)[len(*distances)-1] = dist
		insertSorted(neighbors, distances)
	}

	var first, second *KDTreeNode
	if (node.Axis == 0 && target.X < node.Point.X) || (node.Axis == 1 && target.Y < node.Point.Y) {
		first = node.Left
		second = node.Right
	} else {
		first = node.Right
		second = node.Left
	}

	tree.nearestNeighborsRecursive(first, target, k, neighbors, distances)

	var planeDist float64
	if node.Axis == 0 {
		planeDist = math.Abs(target.X - node.Point.X)
	} else {
		planeDist = math.Abs(target.Y - node.Point.Y)
	}

	if len(*neighbors) < k || planeDist < (*distances)[len(*distances)-1] {
		tree.nearestNeighborsRecursive(second, target, k, neighbors, distances)
	}
}

func euclideanDistance(p1, p2 Point) float64 {
	dx := p1.X - p2.X
	dy := p1.Y - p2.Y
	return math.Sqrt(dx*dx + dy*dy)
}

func insertSorted(neighbors *[]Point, distances *[]float64) {
	n := len(*neighbors)
	for i := n - 1; i > 0; i-- {
		if (*distances)[i] < (*distances)[i-1] {
			(*neighbors)[i], (*neighbors)[i-1] = (*neighbors)[i-1], (*neighbors)[i]
			(*distances)[i], (*distances)[i-1] = (*distances)[i-1], (*distances)[i]
		} else {
			break
		}
	}
}

func TwoOpt(tour []int, distFn DistanceFunc) ([]int, float64) {
	n := len(tour)
	bestTour := make([]int, n)
	copy(bestTour, tour)
	bestLength := tourLength(bestTour, distFn)
	improved := true

	for improved {
		improved = false
		for i := 0; i < n-1; i++ {
			for j := i + 1; j < n; j++ {
				newTour := twoOptSwap(bestTour, i, j)
				newLength := tourLength(newTour, distFn)
				if newLength < bestLength {
					bestTour = newTour
					bestLength = newLength
					improved = true
				}
			}
		}
	}

	return bestTour, bestLength
}

func tourLength(tour []int, distFn DistanceFunc) float64 {
	if len(tour) < 2 {
		return 0
	}
	total := 0.0
	for i := 0; i < len(tour)-1; i++ {
		total += distFn(tour[i], tour[i+1])
	}
	total += distFn(tour[len(tour)-1], tour[0])
	return total
}

func twoOptSwap(tour []int, i, k int) []int {
	n := len(tour)
	newTour := make([]int, n)
	for idx := 0; idx < i; idx++ {
		newTour[idx] = tour[idx]
	}
	for idx := i; idx <= k; idx++ {
		newTour[idx] = tour[k-(idx-i)]
	}
	for idx := k + 1; idx < n; idx++ {
		newTour[idx] = tour[idx]
	}
	return newTour
}

func TwoOptFast(tour []int, distFn DistanceFunc, points []Point, kdtree *KDTree, neighborsCount int) ([]int, float64) {
	n := len(tour)
	bestTour := make([]int, n)
	copy(bestTour, tour)
	bestLength := tourLength(bestTour, distFn)
	improved := true

	posInTour := make([]int, n)
	for i, city := range bestTour {
		posInTour[city] = i
	}

	for improved {
		improved = false
		for i := 0; i < n; i++ {
			city := bestTour[i]
			target := points[city]
			neighbors := kdtree.NearestNeighbors(target, neighborsCount+1)

			for _, nb := range neighbors {
				if nb.ID == city {
					continue
				}
				j := posInTour[nb.ID]
				if j <= i {
					continue
				}

				delta := twoOptDelta(bestTour, i, j, distFn)
				if delta < 0 {
					newTour := twoOptSwap(bestTour, i, j)
					bestTour = newTour
					bestLength += delta
					improved = true

					for idx, c := range bestTour {
						posInTour[c] = idx
					}
				}
			}
		}
	}

	return bestTour, bestLength
}

func twoOptDelta(tour []int, i, k int, distFn DistanceFunc) float64 {
	n := len(tour)
	iPrev := (i - 1 + n) % n
	kNext := (k + 1) % n

	a := distFn(tour[iPrev], tour[i])
	b := distFn(tour[k], tour[kNext])
	c := distFn(tour[iPrev], tour[k])
	d := distFn(tour[i], tour[kNext])

	return c + d - a - b
}

func ThreeOpt(tour []int, distFn DistanceFunc) ([]int, float64) {
	n := len(tour)
	bestTour := make([]int, n)
	copy(bestTour, tour)
	bestLength := tourLength(bestTour, distFn)
	improved := true

	for improved {
		improved = false
		for i := 0; i < n-3; i++ {
			for j := i + 2; j < n-1; j++ {
				for k := j + 2; k < n; k++ {
					newTour, gain := threeOptSwap(bestTour, i, j, k, distFn)
					if gain < 0 {
						bestTour = newTour
						bestLength += gain
						improved = true
					}
				}
			}
		}
	}

	return bestTour, bestLength
}

func threeOptSwap(tour []int, i, j, k int, distFn DistanceFunc) ([]int, float64) {
	n := len(tour)

	seg0 := make([]int, i+1)
	copy(seg0, tour[:i+1])

	seg1 := make([]int, j-i)
	copy(seg1, tour[i+1:j+1])

	seg2 := make([]int, k-j)
	copy(seg2, tour[j+1:k+1])

	seg3 := make([]int, n-k-1)
	copy(seg3, tour[k+1:])

	reverseInts := func(arr []int) {
		for i, j := 0, len(arr)-1; i < j; i, j = i+1, j-1 {
			arr[i], arr[j] = arr[j], arr[i]
		}
	}

	rev1 := make([]int, len(seg1))
	copy(rev1, seg1)
	reverseInts(rev1)

	rev2 := make([]int, len(seg2))
	copy(rev2, seg2)
	reverseInts(rev2)

	candidates := [][]int{
		append(append(append(append([]int{}, seg0...), seg1...), seg2...), seg3...),
		append(append(append(append([]int{}, seg0...), seg2...), seg1...), seg3...),
		append(append(append(append([]int{}, seg0...), rev1...), seg2...), seg3...),
		append(append(append(append([]int{}, seg0...), seg1...), rev2...), seg3...),
		append(append(append(append([]int{}, seg0...), rev2...), seg1...), seg3...),
		append(append(append(append([]int{}, seg0...), seg2...), rev1...), seg3...),
		append(append(append(append([]int{}, seg0...), rev2...), rev1...), seg3...),
	}

	bestGain := 0.0
	bestTour := tour

	for _, cand := range candidates {
		if len(cand) == n {
			gain := tourLength(cand, distFn) - tourLength(tour, distFn)
			if gain < bestGain {
				bestGain = gain
				bestTour = cand
			}
		}
	}

	return bestTour, bestGain
}

func OrOpt(tour []int, distFn DistanceFunc) ([]int, float64) {
	n := len(tour)
	bestTour := make([]int, n)
	copy(bestTour, tour)
	bestLength := tourLength(bestTour, distFn)
	improved := true

	for improved {
		improved = false
		for segmentLen := 1; segmentLen <= 3; segmentLen++ {
			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					if j >= i && j < i+segmentLen {
						continue
					}
					if j+segmentLen > n {
						continue
					}
					newTour := orOptMove(bestTour, i, segmentLen, j)
					if len(newTour) == n {
						newLength := tourLength(newTour, distFn)
						if newLength < bestLength {
							bestTour = newTour
							bestLength = newLength
							improved = true
						}
					}
				}
			}
		}
	}

	return bestTour, bestLength
}

func orOptMove(tour []int, start, length, dest int) []int {
	n := len(tour)
	if start+length > n || dest >= n {
		return tour
	}

	segment := make([]int, length)
	for i := 0; i < length; i++ {
		segment[i] = tour[start+i]
	}

	newTour := make([]int, 0, n)
	for i := 0; i < n; i++ {
		inSegment := false
		for j := 0; j < length; j++ {
			if i == start+j {
				inSegment = true
				break
			}
		}
		if !inSegment {
			newTour = append(newTour, tour[i])
		}
	}

	result := make([]int, 0, n)
	destPos := dest
	if dest > start {
		destPos -= length
	}
	if destPos < 0 {
		destPos = 0
	}
	if destPos > len(newTour) {
		destPos = len(newTour)
	}

	for i := 0; i < destPos; i++ {
		result = append(result, newTour[i])
	}
	result = append(result, segment...)
	for i := destPos; i < len(newTour); i++ {
		result = append(result, newTour[i])
	}

	return result
}
