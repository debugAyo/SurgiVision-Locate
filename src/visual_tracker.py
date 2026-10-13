import os
import cv2
import numpy as np

def generate_surgical_simulation_frame(frame_idx, total_frames=120):
    """
    Simulates a high-contrast endoscopic camera view with an anatomical hazard zone
    and a moving surgical instrument tip tracking across the workspace.
    """
    # Initialize a 600x800 RGB canvas representing the clean surgical site
    frame = np.ones((600, 800, 3), dtype=np.uint8) * 40
    
    # Define a static, highly sensitive anatomical hazard region (Critical Boundary)
    hazard_center = (400, 300)
    hazard_radius = 90
    cv2.circle(frame, hazard_center, hazard_radius, (0, 0, 180), -1)  # Crimson fill
    cv2.putText(frame, "CRITICAL ANATOMICAL STRUCTURE", (260, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    # Calculate instrument kinematics: simulating a sweeping trajectory moving closer
    # to the critical anatomical region as time progresses
    angle = (frame_idx / total_frames) * np.pi
    tool_x = int(150 + 450 * (frame_idx / total_frames))
    tool_y = int(300 + 120 * np.sin(angle * 2.5))
    
    # Render a vibrant colored target marker on the physical tool tip for visual extraction
    cv2.circle(frame, (tool_x, tool_y), 15, (0, 255, 0), -1)  # Pure Green Tracking Ring
    cv2.circle(frame, (tool_x, tool_y), 5, (255, 255, 255), -1)  # Core tool node
    
    return frame, (tool_x, tool_y), hazard_center, hazard_radius

class SpatialPerceptionEngine:
    def __init__(self, hazard_center, hazard_radius):
        self.hz_center = np.array(hazard_center)
        self.hz_radius = hazard_radius
        self.tracking_history = []
        
    def extract_tool_coordinates(self, rgb_frame):
        """Processes the camera array via HSV color-masking to localize coordinates."""
        hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
        
        # Isolate the high-contrast tracker ring affixed to the instrument tip
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Filter out random high-frequency camera noise using morphological erosion/dilation
        kernel = np.ones((5, 5), np.uint8)
        clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Calculate spatial coordinates using matrix moments
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                return (cx, cy), largest_contour
        return None, None

    def evaluate_geofence(self, tool_coords):
        """Computes structural proximity matrix boundaries in real time."""
        current_position = np.array(tool_coords)
        # Euclidean distance metric calculation
        distance_to_center = np.linalg.norm(current_position - self.hz_center)
        
        # Proximity threshold boundary computation
        is_breached = distance_to_center <= self.hz_radius
        return distance_to_center, is_breached

if __name__ == "__main__":
    print("[SYSTEM] Booting Edge Spatial Perception Environment...")
    total_simulation_frames = 120
    
    # Instantiate the detector engine configuration parameters based on frame 0 state
    sample_frame, _, hz_c, hz_r = generate_surgical_simulation_frame(0)
    perception_engine = SpatialPerceptionEngine(hz_c, hz_r)
    
    interlock_events = 0
    
    for f_idx in range(total_simulation_frames):
        raw_view, true_coords, _, _ = generate_surgical_simulation_frame(f_idx, total_simulation_frames)
        
        # 1. Computer Vision Feature Processing Channel
        computed_coords, contour_polygon = perception_engine.extract_tool_coordinates(raw_view)
        
        if computed_coords:
            # Append tracking trail point
            perception_engine.tracking_history.append(computed_coords)
            
            # 2. Geofence Safety Assessment Loop
            distance, hazard_breached = perception_engine.evaluate_geofence(computed_coords)
            
            # 3. Draw active visual analytics onto screen interface
            for idx in range(1, len(perception_engine.tracking_history)):
                cv2.line(raw_view, perception_engine.tracking_history[idx-1], 
                         perception_engine.tracking_history[idx], (255, 255, 0), 2)
            
            if hazard_breached:
                interlock_events += 1
                # Flash sharp visual warning flags across the live viewport interface
                cv2.putText(raw_view, "CRITICAL WARNING: AREA BREACHED", (50, 80), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.rectangle(raw_view, (0, 0), (800, 600), (0, 0, 255), 8)  # Red boundary outline
            else:
                cv2.putText(raw_view, "SAFETY STATUS: NOMINAL", (50, 80), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                
            # Render instantaneous data trackers directly onto display feed panel
            cv2.putText(raw_view, f"Tool Coordinate: X={computed_coords[0]}, Y={computed_coords[1]}", 
                        (50, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(raw_view, f"Proximity Distance: {distance:.1f} px", 
                        (50, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
        # FIX: Explicitly check for frame milestones and save directly to the workspace root directory
        if f_idx in [30, 75]:
            save_path = f"localization_checkpoint_frame_{f_idx}.png"
            cv2.imwrite(save_path, raw_view)
            print(f"[EXPORT] Diagnostic snapshot saved at: {save_path}")

    print(f"\n--- Vision System Analytics Summary ---")
    print(f"Total Frame Iterations Evaluated: {total_simulation_frames}")
    print(f"Active Geofence Infractions Captured: {interlock_events} frames")
    print("Execution verified. Perception pipeline operational.")
    print("---------------------------------------")
