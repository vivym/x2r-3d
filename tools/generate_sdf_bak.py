import io
from pathlib import Path

import torch
import numpy as np
import kaolin.ops.spc as spc_ops
import wisp.ops.mesh as mesh_ops
import wisp.ops.spc as wisp_spc_ops
import zstandard as zstd
from minio import Minio
from tqdm import tqdm

from x2r_3d.accelstructs.octree_as import OctreeAS


def sample_sdf(blas, sample_mode, samples_per_voxel: int = 32):
    # TODO (operel): TBD when kaolin adds a mesh class:
    #  grid is only really needed for filtering out points and more efficient "rand",
    #  better give the mesh as another input and not store the mesh contents in the extent field
    vertices = blas.extra_fields["vertices"]
    faces = blas.extra_fields["faces"]
    level = blas.max_level

    # Here, corners mean "the bottom left corner of the voxel to sample from"
    corners = spc_ops.unbatched_get_level_points(blas.points, blas.pyramid, level)

    # Two pass sampling to figure out sample size
    pts = []
    for mode in sample_mode:
        if mode == "rand":
            # Sample the points.
            pts.append(wisp_spc_ops.sample_spc(corners, level, samples_per_voxel))
    for mode in sample_mode:
        if mode == "rand":
            pass
        elif mode == "near":
            pts.append(mesh_ops.sample_near_surface(vertices.cuda(), faces.cuda(), pts[0].shape[0],
                                                    variance=1.0 / (2 ** level)))
        elif mode == "trace":
            pts.append(mesh_ops.sample_surface(vertices.cuda(), faces.cuda(), pts[0].shape[0])[0])
        else:
            raise Exception(f"Sampling mode {mode} not implemented")

    # Filter out points which do not belong to the narrowband
    pts = torch.cat(pts, dim=0)
    query_results = blas.query(pts, 0)
    pts = pts[query_results.pidx > -1]

    # Sample distances and textures.

    d = mesh_ops.compute_sdf(vertices, faces, pts)
    assert (d.shape[0] == pts.shape[0]), "SDF validation logic failed: the number of returned sdf samples" \
                                         "does not match the number of input coordinates."

    data = dict(
        coords=pts.cpu().numpy(),
        sdf=d.cpu().numpy(),
    )
    return data


def main():
    client = Minio(
        "211.71.15.43",
        access_key="x2r",
        secret_key="x6g2MgkxmfZdJTThXTCb",
    )

    root_path = Path("data/ShapeNetCoreV2-octree")
    for data_path in tqdm(sorted(root_path.glob("*.pth"))):
        blas = OctreeAS.from_dict(octree_dict=torch.load(data_path))
        data = sample_sdf(
            blas=blas,
            sample_mode=["rand", "near", "near", "trace", "trace"],
            samples_per_voxel=32,
        )
        # save to io buffer
        data_bytes = io.BytesIO()
        np.savez_compressed(data_bytes, **data)
        print(f"Compressed size: {data_bytes.getbuffer().nbytes} bytes")

        client.put_object(
            bucket_name="datasets",
            object_name=f"ShapeNetCoreV2/octree/{data_path.stem}.npz",
            data=data_bytes,
            length=-1,
        )


if __name__ == "__main__":
    main()
