"""
MANO hand model fitting for right/left hands.
Loads MANO PKL directly and implements a simplified forward kinematics in PyTorch.
Requires: torch, numpy, scipy, chumpy (needed only to unpickle the MANO PKL).
"""

import os
import pickle
from typing import Dict, Tuple

import numpy as np
import torch


class MANOFitter:
    """
    MANO hand pose estimation and mesh fitting (hand-only model: 778 verts, 16 joints).
    """

    def __init__(self, base_dir: str, device: str = "cpu"):
        """
        Args:
            base_dir: Directory containing MANO/ subfolder with model PKLs.
            device: torch device ("cpu" or "cuda").
        """
        self.base_dir = base_dir
        self.device = torch.device(device)

        # Load both hand models
        self.models = {}
        for hand_type in ["right", "left"]:
            pkl_path = os.path.join(base_dir, "MANO", f"MANO_{hand_type.upper()}.pkl")
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(f"MANO model not found: {pkl_path}")

            with open(pkl_path, "rb") as f:
                data = pickle.load(f, encoding="latin1")

            self.models[hand_type] = self._process_mano_data(data)

        # Cache faces (same for both hands)
        self.faces = self.models["right"]["faces"]

    def _process_mano_data(self, data: dict) -> dict:
        """Convert MANO data dict to torch tensors for hand model."""
        processed = {}

        # v_template: (778, 3)
        processed["v_template"] = torch.from_numpy(
            np.array(data["v_template"], copy=True).reshape(-1, 3)
        ).float().to(self.device)

        # shapedirs: (778, 3, 10)
        shapedirs = np.array(data["shapedirs"], copy=True).reshape(-1, 3, 10)
        processed["shapedirs"] = torch.from_numpy(shapedirs).float().to(self.device)

        # posedirs: hand MANO is (778, 3, 135) for 45 pose parameters
        posedirs = np.array(data["posedirs"], copy=True).reshape(-1, 3, data["posedirs"].shape[-1])
        processed["posedirs"] = torch.from_numpy(posedirs).float().to(self.device)

        # J_regressor: hand MANO is (16, 778)
        J_reg = data["J_regressor"]
        if hasattr(J_reg, "toarray"):
            J_reg = J_reg.toarray()
        processed["J_regressor"] = torch.from_numpy(np.array(J_reg, copy=True)).float().to(self.device)

        # weights: (778, 16)
        weights = data["weights"]
        if hasattr(weights, "toarray"):
            weights = weights.toarray()
        processed["weights"] = torch.from_numpy(np.array(weights, copy=True)).float().to(self.device)

        # faces: (1538, 3)
        processed["faces"] = np.asarray(data["f"], dtype=np.int32)

        return processed

    def forward(
        self,
        pose: torch.Tensor,
        betas: torch.Tensor,
        trans: torch.Tensor,
        hand_type: str = "right",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simplified MANO forward pass (no full LBS): returns vertices and 16 joints.

        Args:
            pose: (B, 45) axis-angle (3 global + 42 joint angles)
            betas: (B, 10) shape coefficients
            trans: (B, 3) translation
            hand_type: "right" or "left"

        Returns:
            verts: (B, 778, 3)
            joints: (B, 16, 3)
        """
        model = self.models[hand_type]
        batch = pose.shape[0]

        # Shape blend
        v_template = model["v_template"].unsqueeze(0)
        shapedirs = model["shapedirs"]
        v_shaped = v_template + torch.einsum("ijk,bk->bij", shapedirs, betas)

        # Pose blend (simplified, uses pose basis directly)
        posedirs = model["posedirs"]
        pose_feat = pose[:, 3:]
        if pose_feat.shape[1] < posedirs.shape[2]:
            pad = torch.zeros(batch, posedirs.shape[2] - pose_feat.shape[1], device=self.device)
            pose_feat = torch.cat([pose_feat, pad], dim=1)
        pose_contrib = torch.einsum("ijk,bk->bij", posedirs, pose_feat)
        v_posed = v_shaped + pose_contrib

        # Regress joints
        J_reg = model["J_regressor"]
        joints = torch.einsum("ij,bjk->bik", J_reg, v_shaped)

        # Simplified: skip per-joint skinning
        verts = v_posed

        # Apply translation
        verts = verts + trans.unsqueeze(1)
        joints = joints + trans.unsqueeze(1)

        return verts, joints

    def fit(
        self,
        landmarks_3d: np.ndarray,
        hand_type: str = "right",
        iters: int = 50,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> Dict:
        """Fit MANO to 21 MediaPipe landmarks by mapping to 16 MANO joints."""
        assert landmarks_3d.shape == (21, 3)
        target = torch.from_numpy(landmarks_3d).float().to(self.device).unsqueeze(0)

        # Map first 16 MediaPipe landmarks (wrist + 5 fingers * 3 joints) to MANO joints
        landmark_indices = list(range(16))
        target_mano = target[:, landmark_indices, :]

        # Parameters
        pose = torch.zeros(1, 45, device=self.device, requires_grad=True)
        betas = torch.zeros(1, 10, device=self.device, requires_grad=True)
        trans = torch.zeros(1, 3, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([pose, betas, trans], lr=lr)

        if verbose:
            print("  Fitting MANO to landmarks...")
        for it in range(iters):
            optimizer.zero_grad()
            verts, joints = self.forward(pose, betas, trans, hand_type=hand_type)
            loss = torch.nn.functional.mse_loss(joints, target_mano)
            loss.backward()
            optimizer.step()
            if verbose and (it + 1) % 20 == 0:
                print(f"    Iter {it + 1}/{iters}: loss={loss.item():.6f}")

        with torch.no_grad():
            verts, joints = self.forward(pose, betas, trans, hand_type=hand_type)

        return {
            "verts": verts[0].cpu().numpy(),
            "mano_joints": joints[0].cpu().numpy(),
            "faces": self.faces,
            "pose": pose.detach().cpu().numpy()[0],
            "betas": betas.detach().cpu().numpy()[0],
            "trans": trans.detach().cpu().numpy()[0],
        }
