using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Microsoft.ML.OnnxRuntime;


namespace ImageRecognizer
{
    public static class ImageRecognizer
    {
        public static string onnxModelPath { get; set; } = @"C:\Users\mary\source\repos3\ImageRecognizer\ImageRecognizer\resnet34-v1-7.onnx";

        private static CancellationTokenSource source { get; set; }
        private static CancellationToken token { get; set; }

        public delegate void OutputHandler(Prediction s);
        public static event OutputHandler ImageRecognizerResultUpdate;

        static ImageRecognizer()
        {
            source = new CancellationTokenSource();
            token = source.Token;
        }

        public static async Task RecognitionAsync(string imagesPath = @"C:\Users\mary\source\repos3\ImageRecognizer\images")
        {
            string[] images = Directory.GetFiles(imagesPath);
            Task[] tasks = new Task[images.Length];

            const int TargetWidth = 224;
            const int TargetHeight = 224;
           
            try {               
                for (int i = 0; i < images.Length; i++)
                {                   
                    tasks[i] = Task.Factory.StartNew((imagePath) =>
                    {
                        using var image = Image.Load<Rgb24>((string)imagePath, out IImageFormat format);

                        image.Mutate(x =>
                        {
                            x.Resize(new ResizeOptions
                            {
                                Size = new Size(TargetWidth, TargetHeight),
                                Mode = ResizeMode.Crop
                            });
                        });

                        var input = new DenseTensor<float>(new[] { 1, 3, TargetHeight, TargetWidth });
                        var mean = new[] { 0.485f, 0.456f, 0.406f };
                        var stddev = new[] { 0.229f, 0.224f, 0.225f };
                        for (int y = 0; y < TargetHeight; y++)
                        {
                            Span<Rgb24> pixelSpan = image.GetPixelRowSpan(y);
                            for (int x = 0; x < TargetWidth; x++)
                            {
                                input[0, 0, y, x] = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
                                input[0, 1, y, x] = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
                                input[0, 2, y, x] = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];
                            }
                        }


                        var inputs = new List<NamedOnnxValue>
                        {
                            NamedOnnxValue.CreateFromTensor("data", input)
                        };

                        // Run inference
                        using var session = new InferenceSession(onnxModelPath);

                        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

                        // Postprocess to get softmax vector
                        IEnumerable<float> output = results.First().AsEnumerable<float>();
                        float sum = output.Sum(x => (float)Math.Exp(x));
                        IEnumerable<float> softmax = output.Select(x => (float)Math.Exp(x) / sum);


                        IEnumerable<Prediction> top1 = softmax.Select((x, i) => new Prediction { Label = LabelMap.Labels[i], Confidence = x })
                                            .OrderByDescending(x => x.Confidence)
                                            .Take(1);

                        Prediction prediction = top1.First();
                        prediction.Path = Path.GetFileName((string)imagePath);

                        if (token.IsCancellationRequested)
                        {
                            throw new OperationCanceledException();
                        }

                        ImageRecognizerResultUpdate?.Invoke(prediction);

                    }, images[i], token);
                } 
                await Task.WhenAll(tasks);
            }
            catch (OperationCanceledException)
            {
                
                Trace.WriteLine(" Users cancelled.");
            }

}
        public static void CancelRecognition()
        {
            source.Cancel();
        }
    }
}
