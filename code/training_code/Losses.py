import torch
import torch.nn as nn
import torch.nn.functional as F

class JSDLoss(nn.Module):
    
    def __init__(self):
        super(JSDLoss, self).__init__()
    
    def forward(self, pred_logits, gt_maps):
        b, c, _, _ = pred_logits.shape
        epsilon = 1e-8
    
        # Flatten spatial dimensions
        flat_pred_logits = pred_logits.view(b, c, -1)
        flat_gt = gt_maps.view(b, c, -1)
        flat_gt_safe = flat_gt + epsilon
        
        # 1. Get P: Ground Truth Probabilities (Normalize Gaussian)
        P = flat_gt_safe / flat_gt_safe.sum(dim=-1, keepdim=True)
        
        # 2. Get Q: Prediction Probabilities
        # Note: We use Softmax (not LogSoftmax) first, because we need to ADD them to calculate M
        Q = F.softmax(flat_pred_logits, dim=-1)
        
        # 3. Calculate M: The Mixture Distribution
        M = 0.5 * (P + Q)
        
        # --- Safe Logs ---
        # Since we added epsilon to P, P > 0 is guaranteed.
        # Q > 0 is guaranteed by Softmax.
        # M > 0 is guaranteed.
        # So log(X) will never be -inf.
        log_P = torch.log(P) 
        log_Q = torch.log(Q + epsilon) 
        log_M = torch.log(M)
        
        # --- Manual Calculation ---
        # We calculate P * (log P - log M) manually.
        # This avoids F.kl_div's internal hidden logic.
        
        # KL(P || M)
        kl_P_M = (P * (log_P - log_M)).sum(dim=-1)
        
        # KL(Q || M)
        kl_Q_M = (Q * (log_Q - log_M)).sum(dim=-1)
        
        loss = 0.5 * (kl_P_M + kl_Q_M).mean()
        
        return loss

class SpatialKLLoss(nn.Module):
    """
    Distribution Matching
    Treats the problem as comparing two probability distributions.
    Best for: Capturing uncertainty and shape of the heatmap.
    """
    def __init__(self):
        super(SpatialKLLoss, self).__init__()

    def forward(self, pred_logits, gt_maps):
        # pred_logits: (B, 10, H, W)
        # gt_maps:     (B, 10, H, W)
        
        b, c, _, _ = pred_logits.shape
        
        # Flatten spatial dimensions
        flat_pred = pred_logits.view(b, c, -1)
        flat_gt = gt_maps.view(b, c, -1)
        
        # 1. Normalize GT to be a valid probability distribution (sum=1)
        # We add 1e-8 to avoid division by zero if a map is empty
        flat_gt_prob = flat_gt / (flat_gt.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 2. Convert Logits to Log-Probabilities
        # log_softmax is numerically more stable than log(softmax(x))
        flat_pred_log_prob = F.log_softmax(flat_pred, dim=-1)
        
        # 3. Calculate KL Divergence
        # reduction='batchmean' averages the loss over the batch size
        loss = F.kl_div(flat_pred_log_prob, flat_gt_prob, reduction='batchmean') + F.kl_div(flat_gt_prob, flat_pred_log_prob, reduction='batchmean')
        
        return loss


class SoftArgmaxLoss(nn.Module):
    """
    Soft-Argmax (DSNT style)
    Extracts coordinates (x,y) from the heatmap and minimizes the L2 distance.
    Best for: High precision coordinate regression.
    """
    def __init__(self, device='cuda'):
        super(SoftArgmaxLoss, self).__init__()
        self.device = device
        self.coord_grid = None # Cache the grid to avoid re-creating it every step

    def _create_meshgrid(self, h, w, device):
        # Create a normalized grid [-1, 1]
        xs = torch.linspace(-1, 1, w, device=device)
        ys = torch.linspace(-1, 1, h, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        # Flatten grid to match the flattened heatmap shape
        return grid_x.reshape(-1), grid_y.reshape(-1)

    def _get_expectation(self, flat_prob, h, w):
        # Create grid if it doesn't exist or shape changed
        if self.coord_grid is None or self.coord_grid[0].shape[0] != h*w:
            self.coord_grid = self._create_meshgrid(h, w, flat_prob.device)
        
        grid_x, grid_y = self.coord_grid
        
        # Expected Value E[x] = sum(prob * x_coord)
        expected_x = torch.sum(flat_prob * grid_x, dim=-1)
        expected_y = torch.sum(flat_prob * grid_y, dim=-1)
        
        # Stack to get (B, 10, 2) -> (x, y) pairs
        return torch.stack([expected_x, expected_y], dim=-1)

    def forward(self, pred_logits, gt_maps):
        b, c, h, w = pred_logits.shape
        
        # 1. Convert Pred logits to Probabilities
        flat_pred = pred_logits.view(b, c, -1)
        pred_prob = F.softmax(flat_pred, dim=-1)
        
        # 2. Convert GT maps to Probabilities (just to be safe)
        flat_gt = gt_maps.view(b, c, -1)
        gt_prob = flat_gt / (flat_gt.sum(dim=-1, keepdim=True) + 1e-8)

        # 3. Calculate Center of Mass (Expectation) for both
        # We extract the "true" coordinates from the GT map dynamically
        pred_coords = self._get_expectation(pred_prob, h, w)
        gt_coords = self._get_expectation(gt_prob, h, w)
        
        # 4. MSE Loss on the coordinates
        loss = F.mse_loss(pred_coords, gt_coords)
        
        return loss