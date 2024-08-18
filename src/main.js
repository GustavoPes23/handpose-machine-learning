import * as tf from "@tensorflow/tfjs";

import { Data } from "./Data";
import { Connections } from "./Connections";

import { trainModel, setupModel, loadModel } from "./model";
import { setupCamera } from "./camera";

const UseCache = false; // Defina como true para coletar novos dados no localstorage

let gestures = Data; // Gestos gravados
let recordingGesture = false;
let gestureLabel = "";
let model;
let featuresLength = 0;

let recordingStartTime;
const recordingDuration = 5000; // 50 segundos
let recordingCount = 0;

const buttonLetters = document.querySelectorAll(".button-letter");

function calculateAngle(startPoint, middlePoint, endPoint) {
  // Vetores entre os pontos
  const vector1 = [middlePoint.x - startPoint.x, middlePoint.y - startPoint.y];
  const vector2 = [endPoint.x - middlePoint.x, endPoint.y - middlePoint.y];

  // Produto escalar dos vetores
  const dotProduct = vector1[0] * vector2[0] + vector1[1] * vector2[1];

  // Magnitudes dos vetores
  const magnitude1 = Math.sqrt(vector1[0] ** 2 + vector1[1] ** 2);
  const magnitude2 = Math.sqrt(vector2[0] ** 2 + vector2[1] ** 2);

  // Verificar se alguma magnitude é zero para evitar divisão por zero
  if (magnitude1 === 0 || magnitude2 === 0) return 0;

  // Cálculo do cosseno do ângulo
  const cosineTheta = dotProduct / (magnitude1 * magnitude2);

  // Garantir que o valor esteja dentro do intervalo válido [-1, 1]
  const clampedCosineTheta = Math.min(1, Math.max(-1, cosineTheta));

  // Cálculo do ângulo em radianos
  const angle = Math.acos(clampedCosineTheta);

  return angle; // Em radianos
}

function extractFeatures(keypoints) {
  const features = [];

  Connections.forEach(([start, end]) => {
    const startPoint = keypoints.find((kp) => kp.name === start);
    const endPoint = keypoints.find((kp) => kp.name === end);
    if (startPoint && endPoint) {
      // Distância
      const distance = Math.sqrt(
        (endPoint.x - startPoint.x) ** 2 + (endPoint.y - startPoint.y) ** 2
      );
      features.push(distance);

      // Ângulo (Se houver ponto médio)
      const middlePoint = keypoints.find(
        (kp) => kp.name === "middle_finger_tip"
      );
      if (middlePoint) {
        const angle = calculateAngle(startPoint, middlePoint, endPoint);
        features.push(angle);
      }
    }
  });

  // Normalização se necessário
  featuresLength = features.length;
  return features;
}

async function detectHands(detector) {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  async function detect() {
    const predictions = await detector.estimateHands(video);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (predictions.length > 0) {
      predictions.forEach(async (prediction) => {
        const keypoints = prediction.keypoints;

        keypoints.forEach((keypoint) => {
          ctx.beginPath();
          ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
          ctx.fillStyle = "red";
          ctx.fill();
        });

        const connections = [
          ["wrist", "thumb_cmc"],
          ["thumb_cmc", "thumb_mcp"],
          ["thumb_mcp", "thumb_ip"],
          ["thumb_ip", "thumb_tip"],
          ["wrist", "index_finger_mcp"],
          ["index_finger_mcp", "index_finger_pip"],
          ["index_finger_pip", "index_finger_dip"],
          ["index_finger_dip", "index_finger_tip"],
          ["wrist", "middle_finger_mcp"],
          ["middle_finger_mcp", "middle_finger_pip"],
          ["middle_finger_pip", "middle_finger_dip"],
          ["middle_finger_dip", "middle_finger_tip"],
          ["wrist", "ring_finger_mcp"],
          ["ring_finger_mcp", "ring_finger_pip"],
          ["ring_finger_pip", "ring_finger_dip"],
          ["ring_finger_dip", "ring_finger_tip"],
          ["wrist", "pinky_finger_mcp"],
          ["pinky_finger_mcp", "pinky_finger_pip"],
          ["pinky_finger_pip", "pinky_finger_dip"],
          ["pinky_finger_dip", "pinky_finger_tip"],
        ];

        connections.forEach(([start, end]) => {
          const startKeypoint = keypoints.find((kp) => kp.name === start);
          const endKeypoint = keypoints.find((kp) => kp.name === end);
          if (startKeypoint && endKeypoint) {
            ctx.beginPath();
            ctx.moveTo(startKeypoint.x, startKeypoint.y);
            ctx.lineTo(endKeypoint.x, endKeypoint.y);
            ctx.strokeStyle = "blue";
            ctx.lineWidth = 2;
            ctx.stroke();
          }
        });

        const features = extractFeatures(keypoints);
        if (features.length <= 0 || !model) {
          return;
        }

        const inputTensor = tf.tensor2d([features], [1, features.length]);
        const modelPrediction = model.predict(inputTensor).arraySync();
        const gestureIndex = modelPrediction[0].indexOf(
          Math.max(...modelPrediction[0])
        );
        const detectedGesture = Object.keys(gestures)[gestureIndex];
        if (!detectedGesture) {
          document.getElementById("output").textContent = "";
        }
        document.getElementById(
          "output"
        ).textContent = `Detected: ${detectedGesture}`;
      });
    }

    requestAnimationFrame(detect);
  }

  detect();
}

function startRecording(label) {
  recordingGesture = true;
  gestureLabel = label;
  if (!gestures[label]) {
    const data = UseCache ? JSON.parse(localStorage.getItem("gestures")) : null;
    gestures[label] = data && data[label] ? data[label] : [];
  }
  document.getElementById("output").textContent = `Recording ${label}`;
  recordingStartTime = Date.now();
  recordingCount = 0; // Reinicia o contador de gravações
  setTimeout(() => {
    recordingGesture = false;
    document.getElementById("output").textContent = "";
  }, recordingDuration); // Continua gravando por 10 segundos
}

async function recordGesture(detector) {
  if (recordingGesture && Date.now() - recordingStartTime < recordingDuration) {
    const video = document.getElementById("video");
    const predictions = await detector.estimateHands(video);

    if (predictions.length <= 0) {
      return;
    }

    const keypoints = predictions[0].keypoints;
    const features = extractFeatures(keypoints);
    if (features.length <= 0) {
      return;
    }

    gestures[gestureLabel].push(features);

    if (UseCache) {
      localStorage.setItem("gestures", JSON.stringify(gestures));
    }

    console.log(`Recorded ${gestureLabel} #${recordingCount + 1}`, features);
    document.getElementById(
      "output"
    ).textContent = `Recorded ${gestureLabel} #${recordingCount + 1}`;
    recordingCount++;
  }
}

function createEventRecord() {
  buttonLetters.forEach((button) => {
    button.addEventListener("click", async () => {
      const id = button.getAttribute("id");
      startRecording(id);

      if (Object.keys(gestures).length > 0) {
        await trainModel(gestures, featuresLength, tf, model);
        document.getElementById("output").textContent = "Model trained!";
      }

      document.getElementById(
        "output"
      ).textContent = `Started recording ${id}. Collecting data...`;
    });
  });
}

async function main() {
  await setupCamera();
  const detector = await setupModel();
  model = await loadModel(tf); // Tente carregar o modelo salvo anteriormente
  detectHands(detector);

  //Intervalo para coletar dados de forma mais eficiente
  setInterval(() => {
    recordGesture(detector);
  }, 200);

  createEventRecord();
}

main();
