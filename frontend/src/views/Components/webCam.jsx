import React, { useRef, useCallback, useState, useEffect } from "react";
import Webcam from "react-webcam";
import * as faceapi from "face-api.js";

import Button from "@mui/material/Button";
import Grid from "@mui/material/Grid";
import Typography from "@mui/material/Typography";

const thresholdPercentFace = 0.2;
const thresholdFaceScore = 0.5;

const WebcamCapture = ({ setImageSrc, onError }) => {
  const webcamRef = useRef(null);
  const detectionIntervalRef = useRef(null);

  const [modelsReady, setModelsReady] = useState(false);
  const [modelError, setModelError] = useState(null);
  const [cameraError, setCameraError] = useState(null);
  const [faceOK, setFaceOK] = useState("Starting camera...");

  const videoConstraints = {
    width: { ideal: 640 },
    height: { ideal: 480 },
    facingMode: "user",
  };

  useEffect(() => {
    let cancelled = false;

    const loadModels = async () => {
      try {
        const MODEL_URI = `${process.env.PUBLIC_URL}/models`;
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URI);
        if (!cancelled) {
          setModelsReady(true);
          setFaceOK("Position your face in the frame");
        }
      } catch (err) {
        console.error("Face detection models failed to load:", err);
        if (!cancelled) {
          setModelError("Face detection unavailable — you can still capture.");
          setModelsReady(true);
          setFaceOK("Ready to capture");
        }
      }
    };

    loadModels();
    return () => {
      cancelled = true;
    };
  }, []);

  const stopDetection = () => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
  };

  const startDetection = useCallback(() => {
    if (modelError || detectionIntervalRef.current) {
      if (modelError) {
        setFaceOK("Ready to capture");
      }
      return;
    }

    detectionIntervalRef.current = setInterval(async () => {
      if (!webcamRef.current?.video) {
        return;
      }

      try {
        const detections = await faceapi.detectAllFaces(
          webcamRef.current.video,
          new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.4 })
        );

        if (detections.length > 1) {
          setFaceOK("Multiple faces detected");
        } else if (detections[0]) {
          const boxArea =
            Math.round(detections[0].box.height) *
            Math.round(detections[0].box.width);
          const imageArea =
            detections[0].imageWidth * detections[0].imageHeight;
          const percentFace = boxArea / imageArea;

          if (percentFace < thresholdPercentFace) {
            setFaceOK("Come closer");
          } else if (detections[0].score < thresholdFaceScore) {
            setFaceOK("Improve lighting or hold still");
          } else {
            setFaceOK("OK");
          }
        } else {
          setFaceOK("No face detected");
        }
      } catch (err) {
        console.error("Face detection error:", err);
      }
    }, 700);
  }, [modelError]);

  useEffect(() => () => stopDetection(), []);

  const handleUserMedia = () => {
    setCameraError(null);
    startDetection();
  };

  const handleUserMediaError = (err) => {
    stopDetection();
    const message =
      err?.name === "NotAllowedError"
        ? "Camera permission denied. Allow camera access in your browser settings."
        : err?.name === "NotFoundError"
        ? "No camera found on this device."
        : "Could not open camera. Try uploading a photo instead.";
    setCameraError(message);
    setFaceOK(message);
    onError?.(message);
  };

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      stopDetection();
      setImageSrc(imageSrc);
    }
  }, [setImageSrc]);

  const canCapture =
    modelsReady && !cameraError && (modelError || faceOK === "OK");

  return (
    <>
      <Grid item>
        <Typography variant="h6" component="div" textAlign="center" sx={{ mb: 1 }}>
          {!modelsReady ? "Loading face detection..." : faceOK}
        </Typography>
        <Webcam
          id="webcam"
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          screenshotQuality={0.85}
          videoConstraints={videoConstraints}
          onUserMedia={handleUserMedia}
          onUserMediaError={handleUserMediaError}
          style={{ width: "100%", borderRadius: 8 }}
        />
      </Grid>
      <Grid item xs={12}>
        <Button
          onClick={capture}
          variant="contained"
          disabled={!canCapture}
          fullWidth
        >
          Capture photo
        </Button>
      </Grid>
    </>
  );
};

export default WebcamCapture;
