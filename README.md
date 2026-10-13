# SurgiVision-Locate: Edge-AI Spatial Perception and Geometric Geofencing for Endoscopic Workspaces

An optimized, real-time Computer Vision (CV) pipeline engineered to track surgical instruments and enforce active spatial safety constraints (virtual fixtures) over critical anatomical hazard zones without requiring cloud or heavy GPU dependency.

## Paradigm
This repository demonstrates an algorithmic approach to spatial perception, tailored for edge deployment. Instead of relying on power-hungry deep learning architectures that introduce inference latency, the system utilizes high-speed color-space conversion, morphological operations, and topological structural matrices to localize coordinates instantly. A deterministic Euclidean geofence evaluator acts as a runtime safety supervisor to trigger immediate visual safety overrides.

## System Pipeline Architecture
- **Color Space Transform:** Converts input frames from standard RGB to HSV (Hue, Saturation, Value) to achieve robust environmental illumination invariance.
- **Morphological Conditioning Unit:** Applies binary threshold masking and structural erosion/dilation transformations to eliminate specular reflections and sensor artifacts.
- **Topological Localization Engine:** Computes instantaneous spatial contours and structural central moments to resolve instrument coordinates (X, Y) in real time.
- **Euclidean Geofence Supervisor:** Calculates continuous vector distance metrics relative to forbidden regions, validating workspace safety thresholds at runtime.

## Verification Checkpoints
The pipeline generates high-resolution tracking logs and snapshots that illustrate safety states. The system accurately differentiates normal adjustments from spatial safety violations:

- **Nominal Tracking State (`localization_checkpoint_frame_30.png`):** Features continuous coordinate calculation and spatial tracking overlays as the instrument moves safely.
- **Active Hazard Interlock (`localization_checkpoint_frame_75.png`):** Shows instant red warning outlines and a critical status alert overlay when safe geometric limits are crossed.
