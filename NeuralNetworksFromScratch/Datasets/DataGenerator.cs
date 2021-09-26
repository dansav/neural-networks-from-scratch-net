using System;
using System.Linq;

namespace NeuralNetworksFromScratch
{
    public static class DataGenerator
    {
        static Random _rnd = new Random(); //reuse this if you are generating many

        public static (float[][] X, int[] y) GenerateSpiralData(int points, int classes)
        {
            var X = Enumerable.Range(0, points * classes).Select(i => new float[] { 0.0f, 0.0f }).ToArray();
            var y = Enumerable.Range(0, points * classes).Select(i => 0).ToArray();

            for (int classNumber = 0; classNumber < classes; classNumber++)
            {
                var ix = Enumerable.Range(points * classNumber, points * (classNumber + 1) - points * classNumber);
                var r = LinSpace(0, 1, points); // radius
                var t = LinSpace(classNumber * 4, (classNumber + 1) * 4, points).Select(x => x + RndGauss() * 0.2f).ToArray();

                int i = 0;
                foreach (var x in ix)
                {
                    X[x][0] = r[i] * MathF.Sin(t[i] * 2.5f);
                    X[x][1] = r[i] * MathF.Cos(t[i] * 2.5f);
                    y[x] = classNumber;
                    i++;
                }
            }

            return (X, y);
        }

        private static float[] LinSpace(float start, float end, int count)
        {
            return Enumerable
                .Range(0, count)
                .Select(i => start + (end - start) * (float)i / (count - 1))
                .ToArray();
        }

        /// <summary>
        /// Obtains normally (Gaussian) distributed random numbers, using the Box-Muller
        /// transformation.  This transformation takes two uniformly distributed deviates
        /// within the unit circle, and transforms them into two independently distributed normal deviates.
        /// </summary>
        /// <returns></returns>
        private static float RndGauss()
        {
            float u1 = 1.0f - _rnd.NextSingle(); //uniform(0,1] random doubles
            float u2 = 1.0f - _rnd.NextSingle();
            return MathF.Sqrt(-2.0f * MathF.Log(u1))
                * MathF.Sin(2.0f * MathF.PI * u2); //random normal(0,1)
        }

        private static float[] RndGauss(int count)
        {
            return Enumerable.Range(0, count).Select(i => RndGauss()).ToArray();
        }
    }
}
