"""Fix a per-joint GPU sync in smplx's linear-blend-skinning (LBS).

`smplx.lbs.batch_rigid_transform` builds the kinematic chain with::

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        ...

`parents` is a CUDA buffer, so `parents[i]` is a 0-d CUDA tensor.  Indexing the
Python list `transform_chain[parents[i]]` therefore calls `.item()` on every one
of the 24 joints of EVERY SMPL forward — ~24 device->host syncs per LBS pass.
Those syncs drain the shared CUDA stream, which is exactly what inflates the
WHAM network from ~69 ms isolated to ~138-167 ms when it shares the GPU with the
detector/extractor threads.

The fix converts `parents` to a cached Python list (one bulk copy, first call
only) and indexes the chain with plain ints — numerically identical, zero syncs.

Applied once at import time by ``lib/models/smpl.py``.
"""
import torch
import torch.nn.functional as F
import smplx.lbs as _lbs


_parents_cache = {}


def _parents_to_list(parents):
    """Convert the (static) parents tensor to a Python list, cached by buffer."""
    key = parents.data_ptr()
    lst = _parents_cache.get(key)
    if lst is None:
        lst = parents.detach().cpu().tolist()
        _parents_cache[key] = lst
    return lst


def _batch_rigid_transform_fast(rot_mats, joints, parents, dtype=torch.float32):
    if torch.is_tensor(parents):
        parents = _parents_to_list(parents)

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = _lbs.transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, len(parents)):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


def patch_smplx_lbs():
    """Replace smplx.lbs.batch_rigid_transform with the sync-free version."""
    if _lbs.batch_rigid_transform is not _batch_rigid_transform_fast:
        _lbs.batch_rigid_transform = _batch_rigid_transform_fast
