using System.Collections;
using System.Collections.Generic;
using System;
using System.Linq;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Rendering;

// Based on https://www.youtube.com/watch?v=ggmArUbRvC4
public class GetInferenceFromModel : MonoBehaviour {
    public RenderTexture streamProjectorTexture;
    public Texture2D runtimeTexture;
    public NNModel modelAsset;

    private Model runtimeModel;

    private IWorker worker; 

    [Serializable]
    public struct Prediction {
        public int predictedValue;
        public float[] predictedArr;

        public void SetPrediction(Tensor t) {
            predictedArr = t.AsFloats();
            predictedValue = Array.IndexOf(predictedArr, predictedArr.Max());   //Argmax
            Debug.Log($" Predicted {predictedValue}");
        }
    }

    public Prediction prediction;

    void Start() {
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(runtimeModel, WorkerFactory.Device.GPU);
        prediction = new Prediction();
    }

    void Update() {
        if (Input.GetKeyDown(KeyCode.Space)) {
            runtimeTexture = RenderTexturetoTexture2D(streamProjectorTexture);
            Predict();
        }
    }

    // Outputs an image based on a render texture. Creates 2 textures in the process, so probably not efficient.
    public Texture2D RenderTexturetoTexture2D(RenderTexture rTex, int sizeX = 28, int sizeY = 28) {
        Texture2D dest = new Texture2D(rTex.width, rTex.height, TextureFormat.RGBA32, false);
        dest.Apply(false);
        Graphics.CopyTexture(rTex, dest);

        Texture2D outTex = new Texture2D(sizeX, sizeY, TextureFormat.RGBA32, false);
        outTex.Apply();
        Graphics.ConvertTexture(dest, outTex);
        return outTex;
    }
    private void Predict() {
        // make a tensor out of a grayscale image
        var channelCount = 1;
        var inputX = new Tensor(runtimeTexture, channelCount);

        Tensor outputY = worker.Execute(inputX).PeekOutput();

        prediction.SetPrediction(outputY);

        //Manage Memory
        inputX.Dispose();
    }

    private void OnDestroy() {
        worker?.Dispose();
    }
}
