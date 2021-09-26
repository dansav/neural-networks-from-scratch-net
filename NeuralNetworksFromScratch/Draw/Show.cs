using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Wpf;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;

namespace NeuralNetworksFromScratch.Draw
{
    public static class Show
    {
        static readonly Thread _guiThread;

        static Show()
        {
            _guiThread = new Thread(() =>
            {
                var app = Application.Current ?? new Application() { ShutdownMode = ShutdownMode.OnExplicitShutdown };
                app.Run();
            });
            _guiThread.SetApartmentState(ApartmentState.STA);
            _guiThread.IsBackground = true;
        }

        public static void Plot(IEnumerable<Series> series, string title)
        {
            if (!_guiThread.IsAlive)
            {
                _guiThread.Start();
                while (Application.Current is null)
                {
                    Console.WriteLine("Waiting for WPF application instance to become available...");
                    Thread.Sleep(500);
                }
            }

            AutoResetEvent windowClosed = new AutoResetEvent(false);

            Application.Current.Dispatcher.Invoke(() =>
            {
                var win = new Window() { Width = 600, Height = 600, WindowStartupLocation = WindowStartupLocation.CenterOwner };
                WindowInteropHelper helper = new WindowInteropHelper(win);
                helper.Owner = NativeMethods.GetConsoleHandle();

                //var text = new System.Windows.Controls.TextBlock()
                //{
                //    Text = "Hello",
                //    FontSize = 50,
                //    HorizontalAlignment = System.Windows.HorizontalAlignment.Center,
                //    VerticalAlignment = System.Windows.VerticalAlignment.Center,
                //};

                var plot = new PlotView();
                win.Content = plot;
                win.Loaded += delegate 
                {
                    plot.Model = new PlotModel() { Title = title };
                    foreach (var s in series)
                    {
                        plot.Model.Series.Add(s);
                    }

                    plot.InvalidatePlot();
                };
                win.Closed += delegate { windowClosed.Set(); };
                win.Show();
            });

            
            windowClosed.WaitOne();
        }

        private static class NativeMethods 
        {
            [DllImport("kernel32.dll", EntryPoint = "GetConsoleWindow", SetLastError = true)]
            public static extern IntPtr GetConsoleHandle();

            [DllImport("user32.dll")]
            public static extern IntPtr SetParent(IntPtr hWndChild, IntPtr hWndNewParent);
        }
    }
}
