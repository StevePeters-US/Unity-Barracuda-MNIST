using System.Collections;
using System.Collections.Generic;
using System;
using System.Linq;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Rendering;
using TMPro;

// Based on https://www.youtube.com/watch?v=ggmArUbRvC4
public class PredictFromRenderTexture : MonoBehaviour {
    public RenderTexture MNISTProjectorTexture; // This texture needs to be generated in MJPEGStreamDecoder.SendFrame()

    private Texture2D runtimeTexture;
    public NNModel modelAsset;

    private Model runtimeModel;

    private IWorker worker;

    public TMP_Text predictionText;

    public bool updateAutomatically = true;
    public float updateFrequency = 0.1f;

    [Serializable]
    public struct Prediction {
        public int predictedValue;
        public float[] predictedArr;
        public void SetPrediction(Tensor t, TMP_Text predictionText) {
            predictedArr = t.AsFloats();
            predictedValue = Array.IndexOf(predictedArr, predictedArr.Max());   //Argmax
            Debug.Log($" Predicted {predictedValue}");
            predictionText.SetText(predictedValue.ToString());
        }
    }

    public Prediction prediction;

    void Start() {
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(runtimeModel, WorkerFactory.Device.GPU);
        prediction = new Prediction();

        if (updateAutomatically) {
            InvokeRepeating("Predict", updateFrequency, updateFrequency);
        }
    }

    void Update() {
        if (!updateAutomatically && Input.GetKeyDown(KeyCode.Space)) {
            Predict();
        }
    }

    // Outputs an image based on a render texture. 
    public Texture2D RenderTexturetoTexture2D(RenderTexture rTex, int sizeX = 28, int sizeY = 28) {
        Texture2D dest = new Texture2D(rTex.width, rTex.height, TextureFormat.R8, false);
        dest.Apply(false);
        Graphics.CopyTexture(rTex, dest);
        return dest;
    }
    private void Predict() {
        runtimeTexture = RenderTexturetoTexture2D(MNISTProjectorTexture);

        // make a tensor out of a grayscale image
        var channelCount = 1;
        var inputX = new Tensor(runtimeTexture, channelCount);

        Tensor outputY = worker.Execute(inputX).PeekOutput();
        prediction.SetPrediction(outputY, predictionText);

        //Manage Memory
        inputX.Dispose();
    }

    private void OnDestroy() {
        worker?.Dispose();
    }
}
