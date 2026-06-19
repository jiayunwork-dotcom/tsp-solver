package output

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"strconv"

	"github.com/tsp-solver/pkg/ga"
	"github.com/tsp-solver/pkg/tsp"
	"github.com/tsp-solver/pkg/tsptw"
)

func WriteConvergenceCSV(history *ga.GAGenerationHistory, filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create CSV file: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	header := []string{"generation", "best_fitness", "avg_fitness", "diversity", "improvement_rate", "stagnation_count"}
	if err := writer.Write(header); err != nil {
		return err
	}

	for i := range history.Generations {
		row := []string{
			strconv.Itoa(history.Generations[i]),
			strconv.FormatFloat(history.BestFitness[i], 'f', 10, 64),
			strconv.FormatFloat(history.AvgFitness[i], 'f', 10, 64),
			strconv.FormatFloat(history.Diversity[i], 'f', 10, 64),
			strconv.FormatFloat(history.ImprovementRate[i], 'f', 6, 64),
			strconv.Itoa(history.StagnationCount[i]),
		}
		if err := writer.Write(row); err != nil {
			return err
		}
	}

	return nil
}

func GenerateTSPVisualization(problem *tsp.TSPProblem, tour []int, outputPath string) error {
	if len(problem.Cities) == 0 {
		return fmt.Errorf("no cities to visualize")
	}

	minX, maxX := problem.Cities[0].X, problem.Cities[0].X
	minY, maxY := problem.Cities[0].Y, problem.Cities[0].Y
	for _, city := range problem.Cities {
		if city.X < minX {
			minX = city.X
		}
		if city.X > maxX {
			maxX = city.X
		}
		if city.Y < minY {
			minY = city.Y
		}
		if city.Y > maxY {
			maxY = city.Y
		}
	}

	margin := 50.0
	width := 800.0
	height := 600.0
	plotWidth := width - 2*margin
	plotHeight := height - 2*margin

	scaleX := plotWidth / (maxX - minX + 1e-10)
	scaleY := plotHeight / (maxY - minY + 1e-10)
	scale := scaleX
	if scaleY < scale {
		scale = scaleY
	}

	offsetX := (width - (maxX-minX)*scale) / 2
	offsetY := (height - (maxY-minY)*scale) / 2

	svgContent := fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="%.0f" height="%.0f" viewBox="0 0 %.0f %.0f">
  <style>
    .city { fill: #e74c3c; stroke: #c0392b; stroke-width: 1.5; }
    .path { stroke: #3498db; stroke-width: 2; fill: none; opacity: 0.8; }
    .start-city { fill: #27ae60; stroke: #1e8449; stroke-width: 2; }
    .label { font-family: Arial, sans-serif; font-size: 10px; fill: #333; }
    .edge-label { font-family: Arial, sans-serif; font-size: 8px; fill: #888; }
    .title { font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; fill: #2c3e50; }
    .info { font-family: Arial, sans-serif; font-size: 12px; fill: #7f8c8d; }
  </style>

  <text x="%.0f" y="30" text-anchor="middle" class="title">TSP Path: %s</text>
  <text x="%.0f" y="50" text-anchor="middle" class="info">Cities: %d | Distance: %.2f</text>
`, width, height, width, height,
		width/2, problem.Name,
		width/2, problem.NumCities, problem.TourLength(tour))

	pathData := ""
	edgeLabels := ""
	n := len(tour)
	for i, cityIdx := range tour {
		city := problem.Cities[cityIdx]
		x := offsetX + (city.X-minX)*scale
		y := offsetY + (maxY-city.Y)*scale
		if i == 0 {
			pathData += fmt.Sprintf("M %.2f %.2f", x, y)
		} else {
			pathData += fmt.Sprintf(" L %.2f %.2f", x, y)
		}

		nextIdx := tour[(i+1)%n]
		nextCity := problem.Cities[nextIdx]
		nx := offsetX + (nextCity.X-minX)*scale
		ny := offsetY + (maxY-nextCity.Y)*scale

		midX := (x + nx) / 2
		midY := (y + ny) / 2

		dist := problem.Distance(cityIdx, nextIdx)

		dx := nx - x
		dy := ny - y
		perpX := -dy
		perpY := dx
		perpLen := 1.0
		if perpX*perpX+perpY*perpY > 1e-10 {
			perpLen = 8.0 / (perpX*perpX + perpY*perpY)
			perpLen = math.Sqrt(perpLen)
		}
		labelX := midX + perpX*perpLen
		labelY := midY + perpY*perpLen

		edgeLabels += fmt.Sprintf(`  <text x="%.2f" y="%.2f" text-anchor="middle" class="edge-label">%.1f</text>`+"\n",
			labelX, labelY, dist)
	}
	firstCity := problem.Cities[tour[0]]
	firstX := offsetX + (firstCity.X-minX)*scale
	firstY := offsetY + (maxY-firstCity.Y)*scale
	pathData += fmt.Sprintf(" L %.2f %.2f Z", firstX, firstY)

	svgContent += fmt.Sprintf(`  <path d="%s" class="path"/>`+"\n", pathData)
	svgContent += edgeLabels

	for i, cityIdx := range tour {
		city := problem.Cities[cityIdx]
		x := offsetX + (city.X-minX)*scale
		y := offsetY + (maxY-city.Y)*scale
		class := "city"
		r := 4.0
		if i == 0 {
			class = "start-city"
			r = 6.0
		}
		svgContent += fmt.Sprintf(`  <circle cx="%.2f" cy="%.2f" r="%.1f" class="%s"/>`+"\n", x, y, r, class)
		svgContent += fmt.Sprintf(`  <text x="%.2f" y="%.2f" text-anchor="middle" class="label">%d</text>`+"\n",
			x, y-10, cityIdx)
	}

	svgContent += "</svg>\n"

	return os.WriteFile(outputPath, []byte(svgContent), 0644)
}

func GenerateTSPTWVisualization(problem *tsptw.TSPTWProblem, tour []int, eval *tsptw.TourEvaluation, outputPath string) error {
	if len(problem.Cities) == 0 {
		return fmt.Errorf("no cities to visualize")
	}

	minX, maxX := problem.Cities[0].X, problem.Cities[0].X
	minY, maxY := problem.Cities[0].Y, problem.Cities[0].Y
	for _, city := range problem.Cities {
		if city.X < minX {
			minX = city.X
		}
		if city.X > maxX {
			maxX = city.X
		}
		if city.Y < minY {
			minY = city.Y
		}
		if city.Y > maxY {
			maxY = city.Y
		}
	}

	margin := 60.0
	width := 900.0
	height := 700.0
	plotWidth := width - 2*margin
	plotHeight := height - 2*margin

	scaleX := plotWidth / (maxX - minX + 1e-10)
	scaleY := plotHeight / (maxY - minY + 1e-10)
	scale := scaleX
	if scaleY < scale {
		scale = scaleY
	}

	offsetX := (width - (maxX-minX)*scale) / 2
	offsetY := (height - (maxY-minY)*scale) / 2

	violatedCities := make(map[int]bool)
	for _, v := range eval.Visits {
		if v.Violation > 1e-10 {
			violatedCities[v.CityID] = true
		}
	}

	svgContent := fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="%.0f" height="%.0f" viewBox="0 0 %.0f %.0f">
  <style>
    .depot { fill: #2980b9; stroke: #1a5276; stroke-width: 2; }
    .city-ok { fill: #27ae60; stroke: #1e8449; stroke-width: 1.5; }
    .city-violated { fill: #e74c3c; stroke: #c0392b; stroke-width: 2; }
    .path { stroke: #3498db; stroke-width: 2; fill: none; opacity: 0.8; }
    .path-return { stroke: #9b59b6; stroke-width: 2; fill: none; opacity: 0.8; stroke-dasharray: 5,3; }
    .path-return-violated { stroke: #e74c3c; stroke-width: 2.5; fill: none; opacity: 0.9; stroke-dasharray: 5,3; }
    .label { font-family: Arial, sans-serif; font-size: 10px; fill: #333; }
    .tw-label { font-family: Arial, sans-serif; font-size: 6px; fill: #888; }
    .travel-label { font-family: Arial, sans-serif; font-size: 8px; fill: #5dade2; }
    .title { font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; fill: #2c3e50; }
    .info { font-family: Arial, sans-serif; font-size: 12px; fill: #7f8c8d; }
    .legend-text { font-family: Arial, sans-serif; font-size: 10px; fill: #333; }
  </style>

  <text x="%.0f" y="30" text-anchor="middle" class="title">TSPTW Path: %s</text>
  <text x="%.0f" y="50" text-anchor="middle" class="info">Cities: %d | Distance: %.2f | Wait: %.2f | Violation: %.2f</text>
`, width, height, width, height,
		width/2, problem.Name,
		width/2, problem.NumCities, eval.TotalDistance, eval.TotalWaitTime, eval.TotalViolation)

	svgContent += `  <circle cx="30" cy="640" r="6" class="depot"/>` + "\n"
	svgContent += `  <text x="42" y="644" class="legend-text">Depot</text>` + "\n"
	svgContent += `  <circle cx="100" cy="640" r="5" class="city-ok"/>` + "\n"
	svgContent += `  <text x="112" y="644" class="legend-text">On time</text>` + "\n"
	svgContent += `  <circle cx="185" cy="640" r="5" class="city-violated"/>` + "\n"
	svgContent += `  <text x="197" y="644" class="legend-text">Violated</text>` + "\n"
	svgContent += `  <line x1="270" y1="640" x2="295" y2="640" class="path"/>` + "\n"
	svgContent += `  <text x="302" y="644" class="legend-text">Travel path</text>` + "\n"
	svgContent += `  <line x1="395" y1="640" x2="420" y2="640" class="path-return"/>` + "\n"
	svgContent += `  <text x="427" y="644" class="legend-text">Return to depot</text>` + "\n"

	n := len(tour)
	pathData := ""
	travelLabels := ""
	for i, cityIdx := range tour {
		city := problem.Cities[cityIdx]
		x := offsetX + (city.X-minX)*scale
		y := offsetY + (maxY-city.Y)*scale
		if i == 0 {
			pathData += fmt.Sprintf("M %.2f %.2f", x, y)
		} else {
			pathData += fmt.Sprintf(" L %.2f %.2f", x, y)
		}

		nextIdx := tour[(i+1)%n]
		if i < n-1 {
			nextCity := problem.Cities[nextIdx]
			nx := offsetX + (nextCity.X-minX)*scale
			ny := offsetY + (maxY-nextCity.Y)*scale

			midX := (x + nx) / 2
			midY := (y + ny) / 2

			travelTime := problem.TravelTime(cityIdx, nextIdx)

			dx := nx - x
			dy := ny - y
			perpX := -dy
			perpY := dx
			perpLen := 1.0
			if perpX*perpX+perpY*perpY > 1e-10 {
				perpLen = 6.0 / math.Sqrt(perpX*perpX + perpY*perpY)
			}
			labelX := midX + perpX*perpLen
			labelY := midY + perpY*perpLen

			travelLabels += fmt.Sprintf(`  <text x="%.2f" y="%.2f" text-anchor="middle" class="travel-label">%.1f</text>`+"\n",
				labelX, labelY, travelTime)
		}
	}

	lastCity := problem.Cities[tour[n-1]]
	lastX := offsetX + (lastCity.X-minX)*scale
	lastY := offsetY + (maxY-lastCity.Y)*scale
	firstCity := problem.Cities[tour[0]]
	firstX := offsetX + (firstCity.X-minX)*scale
	firstY := offsetY + (maxY-firstCity.Y)*scale

	returnTravelTime := problem.TravelTime(tour[n-1], tour[0])
	returnMidX := (lastX + firstX) / 2
	returnMidY := (lastY + firstY) / 2

	dx := firstX - lastX
	dy := firstY - lastY
	perpX := -dy
	perpY := dx
	perpLen := 1.0
	if perpX*perpX+perpY*perpY > 1e-10 {
		perpLen = 6.0 / math.Sqrt(perpX*perpX + perpY*perpY)
	}
	returnLabelX := returnMidX + perpX*perpLen
	returnLabelY := returnMidY + perpY*perpLen

	returnClass := "path-return"
	if eval.ReturnViolation > 1e-10 {
		returnClass = "path-return-violated"
	}

	svgContent += fmt.Sprintf(`  <path d="%s" class="path"/>`+"\n", pathData)
	svgContent += fmt.Sprintf(`  <line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" class="%s"/>`+"\n",
		lastX, lastY, firstX, firstY, returnClass)
	svgContent += travelLabels
	svgContent += fmt.Sprintf(`  <text x="%.2f" y="%.2f" text-anchor="middle" class="travel-label">%.1f</text>`+"\n",
		returnLabelX, returnLabelY, returnTravelTime)

	for i, cityIdx := range tour {
		city := problem.Cities[cityIdx]
		x := offsetX + (city.X-minX)*scale
		y := offsetY + (maxY-city.Y)*scale

		var class string
		var r float64
		if i == 0 {
			class = "depot"
			r = 7.0
		} else if violatedCities[cityIdx] {
			class = "city-violated"
			r = 6.0
		} else {
			class = "city-ok"
			r = 5.0
		}
		svgContent += fmt.Sprintf(`  <circle cx="%.2f" cy="%.2f" r="%.1f" class="%s"/>`+"\n", x, y, r, class)
		svgContent += fmt.Sprintf(`  <text x="%.2f" y="%.2f" text-anchor="middle" class="label">%d</text>`+"\n",
			x, y-12, cityIdx)

		var twText string
		if cityIdx == 0 {
			twText = fmt.Sprintf("Depot [0,%.0f]", city.Latest)
		} else {
			twText = fmt.Sprintf("[%.0f,%.0f]", city.Earliest, city.Latest)
		}
		svgContent += fmt.Sprintf(`  <text x="%.2f" y="%.2f" text-anchor="middle" class="tw-label">%s</text>`+"\n",
			x, y+16, twText)
	}

	svgContent += "</svg>\n"

	return os.WriteFile(outputPath, []byte(svgContent), 0644)
}
