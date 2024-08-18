import * as handPoseDetection from "@tensorflow-models/hand-pose-detection";

const ModelConfig = {
  runtime: "mediapipe",
  solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/hands",
  modelType: "full",
};

function createModel(tf, featuresLength, gestures) {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      units: 128,
      inputShape: [featuresLength],
      kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
      activation: "relu",
    })
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dropout({ rate: 0.3 }));

  model.add(
    tf.layers.dense({
      units: 64,
      kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
      activation: "relu",
    })
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dropout({ rate: 0.3 }));

  model.add(
    tf.layers.dense({
      units: Object.keys(gestures).length,
      activation: "softmax",
    })
  );

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

export async function setupModel() {
  return await handPoseDetection.createDetector(
    handPoseDetection.SupportedModels.MediaPipeHands,
    ModelConfig
  );
}

export async function trainModel(gestures, featuresLength, tf, model) {
  const xs = [];
  const ys = [];
  const labels = Object.keys(gestures);

  labels.forEach((label, index) => {
    gestures[label].forEach((features) => {
      if (features.every((val) => !isNaN(val))) {
        // Verificar se todas as características são números válidos
        xs.push(features);
        const labelArray = Array(labels.length).fill(0);
        labelArray[index] = 1;
        ys.push(labelArray);
      }
    });
  });

  if (xs.length === 0 || ys.length === 0) {
    console.error("No data available for training.");
    return;
  }

  const xsTensor = tf.tensor2d(xs);
  const ysTensor = tf.tensor2d(ys);

  model = createModel(tf, featuresLength, gestures);
  await model.fit(xsTensor, ysTensor, {
    epochs: 30,
    batchSize: 16,
    validationSplit: 0.1,
    callbacks: [
      tf.callbacks.earlyStopping({ monitor: "val_loss", patience: 5 }),
    ],
  });

  // Salvar o modelo após o treinamento
  await model.save("localstorage://gesture-model");
  console.log("Model trained and saved!");
}

export async function loadModel(tf) {
  // Tente carregar o modelo salvo se existir
  try {
    const model = await tf.loadLayersModel("localstorage://gesture-model");
    console.log("Model loaded from local storage!");
    return model;
  } catch (error) {
    console.log("No saved model found. Training a new model.");
  }
}
