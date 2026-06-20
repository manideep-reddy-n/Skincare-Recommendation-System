import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Slider, Alert, Chip, Stack } from "@mui/material";
import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";
import Radio from "@mui/material/Radio";
import RadioGroup from "@mui/material/RadioGroup";
import FormControlLabel from "@mui/material/FormControlLabel";
import FormControl from "@mui/material/FormControl";
import FormLabel from "@mui/material/FormLabel";
import Checkbox from "@mui/material/Checkbox";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import { putForm } from "../controllers/actions";

const skinToneValues = [1, 2, 3, 4, 5, 6];
const skinToneColors = [
  "rgb(33, 28, 40)",
  "rgb(105, 59, 41)",
  "rgb(206, 172, 104)",
  "rgb(240, 227, 171)",
  "rgb(250, 245, 234)",
  "rgb(249, 245, 236)",
];
const skinTypes = ["All", "Oily", "Normal", "Dry"];
const acnes = ["Low", "Moderate", "Severe"];
const otherConcerns = [
  "sensitive",
  "fine lines",
  "wrinkles",
  "redness",
  "pore",
  "pigmentation",
  "blackheads",
  "whiteheads",
  "blemishes",
  "dark circles",
  "eye bags",
  "dark spots",
];

const defaultData = {
  tone: 3,
  type: "Normal",
  acne: "Moderate",
  other_concerns: {},
  confidence: {},
};

const normalizeType = (type) => {
  if (!type) return "Normal";
  if (type === "Oil") return "Oily";
  return type.charAt(0).toUpperCase() + type.slice(1).toLowerCase();
};

const ConfidenceChip = ({ label, value }) => {
  if (!value?.confidence) return null;
  const pct = Math.round(value.confidence * 100);
  const color = value.low_confidence ? "warning" : "success";
  return <Chip size="small" color={color} label={`${label}: ${pct}%`} />;
};

const concernThreshold = 4;

const buildInitialFeatures = (otherConcerns = {}) => {
  const base = {
    normal: false,
    dry: false,
    oily: false,
    combination: false,
    acne: false,
    sensitive: false,
    "fine lines": false,
    redness: false,
    dull: false,
    pore: false,
    pigmentation: false,
    whiteheads: false,
    blemishes: false,
    "dark circles": false,
    "eye bags": false,
  };

  if ((otherConcerns.Wrinkles || 0) >= concernThreshold) base.wrinkles = true;
  if ((otherConcerns.Blackheads || 0) >= concernThreshold) base.blackheads = true;
  if ((otherConcerns.Darkspots || 0) >= concernThreshold) base["dark spots"] = true;
  if ((otherConcerns.Acnes || 0) >= concernThreshold) base.acne = true;

  const problem = otherConcerns.Most_significant_problem;
  const problemMap = {
    Acnes: "acne",
    Blackheads: "blackheads",
    Darkspots: "dark spots",
    Wrinkles: "wrinkles",
  };
  if (problem && problemMap[problem]) {
    base[problemMap[problem]] = true;
  }

  return base;
};

const Form = () => {
  const { state } = useLocation();
  const initial = state?.data ? { ...defaultData, ...state.data } : defaultData;
  initial.type = normalizeType(initial.type);

  const { other_concerns = {}, confidence = {}, warning } = initial;

  const [currType, setCurrType] = useState(initial.type);
  const [currTone, setCurrTone] = useState(parseInt(initial.tone, 10) || 3);
  const [currAcne, setAcne] = useState(initial.acne || "Moderate");
  const [currBlackheads, setBlackheads] = useState(other_concerns.Blackheads || 0);
  const [currDarkspots, setDarkspots] = useState(other_concerns.Darkspots || 0);
  const [currWrinkles, setWrinkles] = useState(other_concerns.Wrinkles || 0);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const [features, setFeatures] = useState(() => buildInitialFeatures(other_concerns));

  const navigate = useNavigate();

  const handleChange = (event) => {
    setFeatures({ ...features, [event.target.name]: event.target.checked });
  };

  const handleSubmit = async () => {
    setSubmitting(true);
    setError(null);

    const payload = { ...features };

    if (currType === "All") {
      payload.normal = true;
      payload.dry = true;
      payload.oily = true;
      payload.combination = true;
    } else {
      payload[currType.toLowerCase()] = true;
    }

    payload.acne = currAcne !== "Low" || (other_concerns.Acnes || 0) >= concernThreshold;
    payload.blackheads = currBlackheads > 0;
    payload["dark spots"] = currDarkspots > 0;
    payload.wrinkles = currWrinkles > 0;

    for (const key of Object.keys(payload)) {
      payload[key] = payload[key] ? 1 : 0;
    }

    try {
      await putForm(payload, currType, currTone, currAcne, navigate);
    } catch (err) {
      setError(err.message || "Failed to get recommendations.");
      setSubmitting(false);
    }
  };

  const lowConfidence =
    confidence?.skin_type?.low_confidence || confidence?.acne?.low_confidence;

  return (
    <Container maxWidth="xs" sx={{ marginTop: "2vh", marginBottom: "4vh" }}>
      <Typography variant="h5" textAlign="center">
        Your Skin Analysis
      </Typography>
      <Typography variant="body2" color="text.secondary" textAlign="center" sx={{ mt: 1 }}>
        Review and adjust the detected values before getting recommendations.
      </Typography>

      {warning && (
        <Alert severity="info" sx={{ mt: 2 }}>
          {warning}
        </Alert>
      )}

      {lowConfidence && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          Some predictions had low confidence. Please verify the values below.
        </Alert>
      )}

      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mt: 2 }}>
        <ConfidenceChip label="Skin type" value={confidence.skin_type} />
        <ConfidenceChip label="Acne" value={confidence.acne} />
        {confidence.skin_tone?.confidence && (
          <Chip
            size="small"
            color="success"
            label={`Tone: ${Math.round(confidence.skin_tone.confidence * 100)}%`}
          />
        )}
      </Stack>

      <FormControl sx={{ marginTop: "3vh", width: "100%" }}>
        <Grid container spacing={2}>
          <Grid item xs={9}>
            <InputLabel>Skin tone (1–6)</InputLabel>
            <Select value={currTone} onChange={(e) => setCurrTone(e.target.value)} fullWidth>
              {skinToneValues.map((value) => (
                <MenuItem key={value} value={value}>
                  Type {value}
                </MenuItem>
              ))}
            </Select>
          </Grid>
          <Grid item xs={3}>
            <div
              style={{
                height: "3rem",
                width: "3rem",
                backgroundColor: skinToneColors[currTone - 1],
                margin: "0 auto",
                borderRadius: "10%",
                border: "1px solid #ccc",
              }}
            />
          </Grid>
        </Grid>

        <Grid marginTop="2vh">
          <FormLabel>Skin type</FormLabel>
          <RadioGroup row value={currType} onChange={(e) => setCurrType(e.target.value)}>
            {skinTypes.map((type) => (
              <FormControlLabel key={type} value={type} control={<Radio />} label={type} />
            ))}
          </RadioGroup>
        </Grid>

        <Grid marginTop="2vh">
          <FormLabel>Acne level</FormLabel>
          <RadioGroup row value={currAcne} onChange={(e) => setAcne(e.target.value)}>
            {acnes.map((ac) => (
              <FormControlLabel key={ac} value={ac} control={<Radio />} label={ac} />
            ))}
          </RadioGroup>
        </Grid>

        <Grid marginTop="2vh">
          <FormLabel>Blackheads</FormLabel>
          <Slider value={currBlackheads} onChange={(e, val) => setBlackheads(val)} step={1} marks min={0} max={10} valueLabelDisplay="auto" />
        </Grid>

        <Grid marginTop="2vh">
          <FormLabel>Dark spots</FormLabel>
          <Slider value={currDarkspots} onChange={(e, val) => setDarkspots(val)} step={1} marks min={0} max={10} valueLabelDisplay="auto" />
        </Grid>

        <Grid marginTop="2vh">
          <FormLabel>Wrinkles</FormLabel>
          <Slider value={currWrinkles} onChange={(e, val) => setWrinkles(val)} step={1} marks min={0} max={10} valueLabelDisplay="auto" />
        </Grid>

        <Grid marginTop="2vh">
          <FormLabel>Other concerns</FormLabel>
          <Grid container>
            {otherConcerns
              .filter((c) => !["wrinkles", "blackheads", "dark spots"].includes(c))
              .map((concern) => (
                <Grid item xs={6} key={concern}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={features[concern]}
                        onChange={handleChange}
                        name={concern}
                      />
                    }
                    label={concern.charAt(0).toUpperCase() + concern.slice(1)}
                  />
                </Grid>
              ))}
          </Grid>
        </Grid>

        <Grid marginTop="2vh">
          <Typography variant="body2" fontWeight="bold">
            Most significant concern: {other_concerns?.Most_significant_problem || "None"}
          </Typography>
        </Grid>

        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}

        <Grid marginTop="2vh">
          <Button onClick={handleSubmit} variant="contained" fullWidth disabled={submitting}>
            {submitting ? "Finding products..." : "Get recommendations"}
          </Button>
        </Grid>
      </FormControl>
    </Container>
  );
};

export default Form;
