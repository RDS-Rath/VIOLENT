import numpy as np

def vif_block_hist(flow):
    flow_vec = flow.flatten()
    count, _ = np.histogram(flow_vec, bins=np.arange(0, 1.05, 0.05))
    block_hist = count / np.sum(count)
    return block_hist

def vif_create_block_hist(flow, N, M):
    hight, width = flow.shape[:2]

    B_hight = (hight - 11) // N
    B_width = (width - 11) // M

    frame_hist = []
    for y in range(6, hight - B_hight - 5, B_hight):
        for x in range(6, width - B_width - 5, B_width):
            block_hist = vif_block_hist(flow[y:y + B_hight, x:x + B_width])
            frame_hist.append(block_hist)
    
    frame_hist = np.concatenate(frame_hist, axis=0)
    return frame_hist
