import React, { useState, useEffect, useRef } from "react";
import { UploadImage } from "../controllers/actions";
import { useNavigate } from "react-router-dom";

import WebcamCapture from "./Components/webCam";

import {
  Grid,
  Container,
  Button,
  Typography,
  Box,
  IconButton,
  Stack,
  useTheme,
  CircularProgress,
  Alert,
} from "@mui/material";
import PhotoCameraIcon from "@mui/icons-material/PhotoCamera";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";

function ImageInput() {
  const [landingPage, setLandingPage] = useState(true);
  const [imageSrc, setImageSrc] = useState(null);
  const [preview, setPreview] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const uploadedRef = useRef(false);
  const navigate = useNavigate();
  const theme = useTheme();

  const resetUpload = () => {
    uploadedRef.current = false;
    setImageSrc(null);
    setPreview(null);
    setUploading(false);
    setError(null);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith("image")) {
      setError(null);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImageSrc(reader.result);
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
    e.target.value = "";
  };

  useEffect(() => {
    if (!imageSrc || uploadedRef.current) {
      return;
    }

    uploadedRef.current = true;
    setUploading(true);
    setError(null);

    UploadImage(imageSrc, navigate).catch((err) => {
      const message =
        err.name === "AbortError"
          ? "Analysis timed out. Please try a smaller photo or restart the backend."
          : err.message || "Failed to analyze image.";
      setError(message);
      setUploading(false);
      uploadedRef.current = false;
    });
  }, [imageSrc, navigate]);

  return (
    <Container
      maxWidth="xs"
      sx={{ minHeight: "100vh", display: "flex", alignItems: "center" }}
    >
      <Grid container spacing={2} justifyContent="center" alignItems="center">
        {uploading ? (
          <Grid item xs={12} textAlign="center">
            <CircularProgress sx={{ mb: 2 }} />
            <Typography variant="h6">Analyzing your photo...</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              This usually takes 15–30 seconds.
            </Typography>
          </Grid>
        ) : landingPage ? (
          <Grid item xs={12} textAlign="center">
            <PhotoCameraIcon
              sx={{ fontSize: "6rem", color: theme.palette.primary.main }}
            />
            <Typography variant="h5" sx={{ mt: 2, mb: 3 }}>
              Choose Image Source
            </Typography>
            {error && (
              <Alert severity="error" sx={{ mb: 2, textAlign: "left" }}>
                {error}
              </Alert>
            )}
            <Stack spacing={2}>
              <Button
                variant="contained"
                startIcon={<PhotoCameraIcon />}
                onClick={() => {
                  setError(null);
                  setLandingPage(false);
                }}
              >
                Take a Photo
              </Button>
              <Button
                variant="outlined"
                component="label"
                startIcon={<UploadFileIcon />}
              >
                Upload from Device
                <input
                  type="file"
                  hidden
                  accept="image/*"
                  onChange={handleFileChange}
                />
              </Button>
            </Stack>
          </Grid>
        ) : (
          <>
            <Grid item xs={12}>
              <IconButton
                onClick={() => {
                  resetUpload();
                  setLandingPage(true);
                }}
              >
                <ArrowBackIcon />
              </IconButton>
            </Grid>
            <Grid item xs={12}>
              <WebcamCapture
                setImageSrc={setImageSrc}
                onError={(message) => setError(message)}
              />
            </Grid>
            {error && (
              <Grid item xs={12}>
                <Alert severity="error">{error}</Alert>
              </Grid>
            )}
          </>
        )}

        {preview && !uploading && (
          <Grid item xs={12} textAlign="center">
            <Typography variant="body1" sx={{ mt: 2 }}>
              Image Preview
            </Typography>
            <Box
              component="img"
              src={preview}
              alt="preview"
              sx={{ width: "100%", borderRadius: 2, mt: 1 }}
            />
            {error && (
              <Button sx={{ mt: 2 }} variant="outlined" onClick={resetUpload}>
                Try Again
              </Button>
            )}
          </Grid>
        )}
      </Grid>
    </Container>
  );
}

export default ImageInput;
