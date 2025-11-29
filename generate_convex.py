"""Generate convex-hull STL meshes from existing STL models.

使用 SciPy 的 ConvexHull 对原始 STL 点云做凸包，然后写回新的 STL。
默认会在文件名后加上后缀 "_convex.stl"，避免覆盖原文件。
"""

import os
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from stl import mesh


def generate_convex_stl(input_stl_path: str, output_stl_path: str) -> None:
    """从 input_stl_path 读取 STL，生成凸包并写入 output_stl_path。"""

    input_stl_path = str(input_stl_path)
    output_stl_path = str(output_stl_path)

    if not os.path.isfile(input_stl_path):
        print(f"[WARN] File not found, skip: {input_stl_path}")
        return

    original_mesh = mesh.Mesh.from_file(input_stl_path)
    print(
        f"[INFO] Loaded {input_stl_path} with {len(original_mesh.vectors)} triangles."
    )

    # 所有三角面片的顶点展平成点云，并去重
    vertices = np.unique(original_mesh.vectors.reshape(-1, 3), axis=0)

    if len(vertices) < 4:
        print(f"[WARN] Not enough vertices to form a 3D convex hull: {input_stl_path}")
        return

    try:
        hull = ConvexHull(vertices)
    except Exception as e:
        print(f"[ERROR] Failed to compute convex hull for {input_stl_path}: {e}")
        return

    # hull.simplices 是三角面顶点索引（在 3D 情况下）
    convex_triangles = hull.simplices
    convex_mesh = mesh.Mesh(np.zeros(len(convex_triangles), dtype=mesh.Mesh.dtype))
    for i, simplex in enumerate(convex_triangles):
        convex_mesh.vectors[i] = vertices[simplex]

    os.makedirs(os.path.dirname(output_stl_path), exist_ok=True)
    convex_mesh.save(output_stl_path)
    print(f"[INFO] Convex hull saved to {output_stl_path}")


def batch_convex_hull_generation(
    input_directory: str,
    output_directory: str | None = None,
    suffix: str = ".stl.convex",
) -> None:
    """批量为目录下所有 STL 生成凸包版本。

    - input_directory: 原始 STL 所在目录
    - output_directory: 输出目录；如果为 None，则与输入目录相同
    - suffix: 输出文件名后缀（在原文件名基础上追加）
    """

    input_dir = Path(input_directory)
    if output_directory is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_directory)

    output_dir.mkdir(parents=True, exist_ok=True)

    stl_files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() == ".stl")
    if not stl_files:
        print(f"[WARN] No STL files found in {input_dir}")
        return

    print(f"[INFO] Found {len(stl_files)} STL files in {input_dir}")

    for in_path in stl_files:
        stem = in_path.stem
        out_name = f"{stem}{suffix}.stl"
        out_path = output_dir / out_name
        generate_convex_stl(str(in_path), str(out_path))


if __name__ == "__main__":
    # 默认对 so101 的 STL 零件批量生成凸包版本
    default_dir = Path("assets/robots/so101/so101_new/assets")
    batch_convex_hull_generation(str(default_dir))
