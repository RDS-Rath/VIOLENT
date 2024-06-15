import cv2
import numpy as np


def vif_create_frame_flow(Prev_F, Current_F, N, M):
    hight, width = Current_F.shape
    
    B_hight = (hight - 11) // N
    B_width = (width - 11) // M
    
    flow = cv2.calcOpticalFlowFarneback(
        Prev_F, Current_F, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    if flow is None or flow.shape[:2] != Current_F.shape:
        raise ValueError("Failed to calculate optical flow or shape mismatch.")
    
    vx = flow[..., 0]
    vy = flow[..., 1]
    flow_magnitude = np.sqrt(vx ** 2 + vy ** 2)
    
    if flow_magnitude.shape[1] != width:
        flow_magnitude = np.pad(flow_magnitude, ((0, 0), (0, 1)), 'constant', constant_values=0)
    elif flow_magnitude.shape[1] > width:
        flow_magnitude = flow_magnitude[:, :width]
    
    return flow_magnitude, vx, vy
