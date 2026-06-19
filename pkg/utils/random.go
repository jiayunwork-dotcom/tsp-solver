package utils

import (
	"math/rand"
	"sync"
	"time"
)

var (
	rng  *rand.Rand
	once sync.Once
	mu   sync.Mutex
)

func ensureInit() {
	once.Do(func() {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	})
}

func GetRand() *rand.Rand {
	ensureInit()
	return rng
}

func RandInt(min, max int) int {
	mu.Lock()
	defer mu.Unlock()
	ensureInit()
	return min + rng.Intn(max-min+1)
}

func RandFloat() float64 {
	mu.Lock()
	defer mu.Unlock()
	ensureInit()
	return rng.Float64()
}

func RandPerm(n int) []int {
	mu.Lock()
	defer mu.Unlock()
	ensureInit()
	return rng.Perm(n)
}

func Shuffle(arr []int) {
	mu.Lock()
	defer mu.Unlock()
	ensureInit()
	rng.Shuffle(len(arr), func(i, j int) {
		arr[i], arr[j] = arr[j], arr[i]
	})
}

func Gaussian(mean, std float64) float64 {
	mu.Lock()
	defer mu.Unlock()
	ensureInit()
	return mean + rng.NormFloat64()*std
}

func SetSeed(seed int64) {
	mu.Lock()
	defer mu.Unlock()
	rng = rand.New(rand.NewSource(seed))
	once = sync.Once{}
}
