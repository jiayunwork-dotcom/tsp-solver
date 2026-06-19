package output

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"github.com/tsp-solver/pkg/ga"
	"github.com/tsp-solver/pkg/tsp"
)

func WriteConvergenceCSV(history *ga.GAGenerationHistory, filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create CSV file: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	header := []string{"generation", "best_fitness", "avg_fitness", "diversity"}
	if err := writer.Write(header); err != nil {
		return err
	}

	for i := range history.Generations {
		row := []string{
			strconv.Itoa(history.Generations[i]),
			strconv.FormatFloat(history.BestFitness[i], 'f', 10, 64),
			strconv.FormatFloat(history.AvgFitness[i], 'f', 10, 64),
			strconv.FormatFloat(history.Diversity[i], 'f', 10, 64),
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
    .title { font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; fill: #2c3e50; }
    .info { font-family: Arial, sans-serif; font-size: 12px; fill: #7f8c8d; }
  </style>

  <text x="%.0f" y="30" text-anchor="middle" class="title">TSP Path: %s</text>
  <text x="%.0f" y="50" text-anchor="middle" class="info">Cities: %d | Distance: %.2f</text>
`, width, height, width, height,
		width/2, problem.Name,
		width/2, problem.NumCities, problem.TourLength(tour))

	pathData := ""
	for i, cityIdx := range tour {
		city := problem.Cities[cityIdx]
		x := offsetX + (city.X-minX)*scale
		y := offsetY + (maxY-city.Y)*scale
		if i == 0 {
			pathData += fmt.Sprintf("M %.2f %.2f", x, y)
		} else {
			pathData += fmt.Sprintf(" L %.2f %.2f", x, y)
		}
	}
	firstCity := problem.Cities[tour[0]]
	firstX := offsetX + (firstCity.X-minX)*scale
	firstY := offsetY + (maxY-firstCity.Y)*scale
	pathData += fmt.Sprintf(" L %.2f %.2f Z", firstX, firstY)

	svgContent += fmt.Sprintf(`  <path d="%s" class="path"/>`+"\n", pathData)

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
