using OxyPlot;
using OxyPlot.Series;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworksFromScratch.Draw
{
    public static class Chart
    {
        public static IEnumerable<ScatterSeries> CreateSeries(float[][] X, int[] y)
        {
            var w = X
                .Zip(y)
                .GroupBy(z => z.Second);

            foreach (var item in w)
            {
                var series = new ScatterSeries()
                {
                    MarkerType = MarkerType.Circle,
                    MarkerSize = 5,
                    Title = $"Class {item.Key}"
                };
                foreach (var point in item.Select(i => i.First))
                {
                    series.Points.Add(new ScatterPoint(point[0], point[1], 5));
                }

                yield return series;
            }
        }
    }
}
