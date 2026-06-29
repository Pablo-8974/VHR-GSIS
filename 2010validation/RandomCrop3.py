import os
import random
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.ops import unary_union


def random_point_in_geom(geom, rng, max_tries=20000):
    """
    在给定(多)边形 geom 内随机采样一个点（拒绝采样）。
    geom: shapely geometry (Polygon/MultiPolygon)
    rng: random.Random
    """
    minx, miny, maxx, maxy = geom.bounds
    for _ in range(max_tries):
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        p = Point(x, y)
        if geom.contains(p):
            return p
    raise RuntimeError("在几何范围内采样点失败：可能几何过小/过窄，或 max_tries 不够。")


def point_window_within_raster(src, x, y, patch_size):
    """
    判断以(x,y)为中心的 patch_size×patch_size 窗口是否完全落在影像内部。
    返回: (ok, window, (row, col))
    """
    half = patch_size // 2
    row, col = src.index(x, y)  # row, col are ints

    # 注意 Window(col_off, row_off, width, height)
    col_off = col - half
    row_off = row - half

    ok = (col_off >= 0) and (row_off >= 0) and \
         (col_off + patch_size <= src.width) and \
         (row_off + patch_size <= src.height)

    if not ok:
        return False, None, (row, col)

    win = Window(col_off=col_off, row_off=row_off, width=patch_size, height=patch_size)
    return True, win, (row, col)


def sample_points(src, geom_target, n_in, n_out, patch_size, seed=42):
    """
    采样 n_in 个目标范围内点 + n_out 个目标范围外但影像范围内点。
    同时保证裁切窗口不会越界。
    """
    rng = random.Random(seed)

    raster_geom = box(*src.bounds)  # shapely box(minx, miny, maxx, maxy)

    # 目标范围裁切到影像范围内，避免 shp 超出影像导致采样失败
    geom_in = geom_target.intersection(raster_geom)
    if geom_in.is_empty:
        raise ValueError("目标范围(状态0/1)与影像范围没有交集，无法采样范围内点。")

    # 影像范围内但不在目标范围的区域
    geom_out = raster_geom.difference(geom_target)
    if geom_out.is_empty:
        raise ValueError("影像范围完全被目标范围覆盖，无法采样范围外点。")

    inside_points = []
    outside_points = []

    # 采样函数（带越界检查的拒绝采样）
    def sample_valid_point(geom, label, max_tries=50000):
        for _ in range(max_tries):
            try:
                p = random_point_in_geom(geom, rng)
                ok, win, (row, col) = point_window_within_raster(src, p.x, p.y, patch_size)
                if ok:
                    return p, win, row, col
            except:
                pass

        raise RuntimeError(f"采样 {label} 点失败：可能因为靠边区域过多，导致1024裁切总是越界。")

    # 目标范围内点
    for i in range(n_in):
        p, win, row, col = sample_valid_point(geom_in, label="inside")
        inside_points.append((p, win, row, col))

    # 目标范围外点
    for i in range(n_out):
        p, win, row, col = sample_valid_point(geom_out, label="outside")
        outside_points.append((p, win, row, col))

    return inside_points, outside_points


def crop_and_save(src, win, out_path):
    """
    从 src 按窗口 win 裁切并保存为 GeoTIFF。
    """
    data = src.read(window=win)  # shape: (bands, H, W)
    meta = src.meta.copy()
    meta.update({
        "height": int(win.height),
        "width": int(win.width),
        "transform": window_transform(win, src.transform)
    })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(data)


def fix_geometry_series(gser):
    """
    修复一批 geometry，尽最大努力让 union 可用：
    1) 过滤空/None
    2) make_valid（Shapely 2.x）或 validation.make_valid（Shapely 1.8）
    3) buffer(0) 作为兜底
    4) 只保留 Polygon/MultiPolygon（如果你的 shp 应该是面）
    """
    import shapely
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
    from shapely.ops import unary_union

    gser = gser.copy()
    gser = gser[gser.notnull()]
    gser = gser[~gser.is_empty]

    # --- 1) make_valid ---
    def _make_valid(g):
        if g is None or g.is_empty:
            return g
        try:
            # shapely 2.x
            from shapely import make_valid
            return make_valid(g)
        except Exception:
            # shapely 1.8.x
            try:
                from shapely.validation import make_valid
                return make_valid(g)
            except Exception:
                return g

    gser = gser.apply(_make_valid)

    # --- 2) buffer(0) 兜底（对自相交有时有效） ---
    def _buffer0(g):
        if g is None or g.is_empty:
            return g
        try:
            if not g.is_valid:
                return g.buffer(0)
            return g
        except Exception:
            return g

    gser = gser.apply(_buffer0)

    # --- 3) 只保留面要素（Polygon/MultiPolygon），必要时从 GeometryCollection 中提取面 ---
    def _keep_polygons(g):
        if g is None or g.is_empty:
            return None
        if isinstance(g, (Polygon, MultiPolygon)):
            return g
        if isinstance(g, GeometryCollection):
            polys = [gg for gg in g.geoms if isinstance(gg, (Polygon, MultiPolygon)) and (not gg.is_empty)]
            if len(polys) == 0:
                return None
            # 合并集合内的面
            try:
                return unary_union(polys)
            except Exception:
                return polys[0]
        return None

    gser = gser.apply(_keep_polygons)
    gser = gser[gser.notnull()]
    gser = gser[~gser.is_empty]

    # 最后再过滤一次 validity（避免 union 再炸）
    gser = gser[gser.is_valid]
    return gser

def main(
    tif_path,
    shp_path,
    out_dir,
    patch_size=1024,
    n_in=5,
    n_out=5,
    status_field="status",
    target_values=(0, 1),
    seed=42
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) 打开影像
    with rasterio.open(tif_path) as src:
        # 2) 读取矢量并投影到影像 CRS
        gdf = gpd.read_file(shp_path)

        if status_field not in gdf.columns:
            raise KeyError(f"shp 中找不到字段 '{status_field}'，现有字段：{list(gdf.columns)}")

        # 过滤 status 为 0/1
        gdf_target = gdf[gdf[status_field].isin(list(target_values))].copy()
        # gdf_target = gdf[gdf[status_field] == 0].copy()
        if len(gdf_target) == 0:
            raise ValueError(f"shp 中 status ∈ {target_values} 的要素为空。")

        # 统一到影像 CRS
        if gdf_target.crs is None:
            raise ValueError("shp 缺少 CRS 信息（gdf.crs is None），请先为 shp 定义正确坐标系。")

        if src.crs is None:
            raise ValueError("tif 缺少 CRS 信息（src.crs is None），请确认输入是带地理参考的 GeoTIFF。")

        if gdf_target.crs != src.crs:
            gdf_target = gdf_target.to_crs(src.crs)

        # 3) 目标范围并集（0/1）

        # 修复几何
        gdf_target = gdf_target[~gdf_target.geometry.is_empty & gdf_target.geometry.notnull()].copy()
        gdf_target["geometry"] = fix_geometry_series(gdf_target.geometry)

        if len(gdf_target) == 0:
            raise ValueError("修复后目标范围为空：请检查 status=0/1 的要素是否都是坏几何或不在影像范围内。")

        # 再 union
        geom_target = unary_union(list(gdf_target.geometry))

        if geom_target.is_empty:
            raise ValueError("目标范围 unary_union 为空，无法继续。")

        # 4) 采样点（确保裁切不越界）
        inside, outside = sample_points(
            src=src,
            geom_target=geom_target,
            n_in=n_in,
            n_out=n_out,
            patch_size=patch_size,
            seed=seed
        )

        # 5) 裁切并保存
        records = []
        idx = 0

        for group_name, pts in [("inside", inside), ("outside", outside)]:
            for (p, win, row, col) in pts:
                idx += 1
                out_name = f"patch_{idx:02d}_{group_name}.tif"
                out_path = os.path.join(out_dir, out_name)

                crop_and_save(src, win, out_path)

                records.append({
                    "id": idx,
                    "group": group_name,
                    "x": p.x,
                    "y": p.y,
                    "row": row,
                    "col": col,
                    "col_off": int(win.col_off),
                    "row_off": int(win.row_off),
                    "patch_size": patch_size,
                    "out_tif": out_name
                })

        # 输出点位信息
        df = pd.DataFrame(records)
        csv_path = os.path.join(out_dir, "sample_points.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 也可输出点shp（可选）
        # point_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs=src.crs)
        # point_gdf.to_file(os.path.join(out_dir, "sample_points.shp"), encoding="utf-8")

        print(f"完成！已输出 {len(records)} 张 patch 到：{out_dir}")
        print(f"采样点记录 CSV：{csv_path}")


def one_step():
    city_dir = '/intelnvme04/jiang.mingyu/slum/GoogleMap'
    save_dir = os.path.join(city_dir, '2010validation')
    os.makedirs(save_dir, exist_ok=True)

    city_dir_list = os.listdir(city_dir)
    for city_dir_name in city_dir_list:
        if 'drop' in city_dir_name:
            continue

        city_dir_path = os.path.join(city_dir, city_dir_name)
        if not os.path.isdir(city_dir_path):
            continue

        city_dir_structure = os.listdir(city_dir_path)
        if '2010' not in city_dir_structure:
            continue

        image_path = os.path.join(city_dir_path, '2010', 'L19.tif')
        if os.path.isfile(image_path):
            _save_dir = os.path.join(save_dir, city_dir_name)

            shp_path = os.path.join(city_dir_path, 'TemporalChange.shp')
            if not os.path.isfile(shp_path):
                continue

            main(
                tif_path=image_path,
                shp_path=shp_path,
                out_dir=_save_dir,
                patch_size=1024,
                n_in=5,
                n_out=5,
                status_field="status",
                target_values=(0, 1),
                seed=2026
            )
            print("保存完成：", image_path)


if __name__ == "__main__":
    # # ====== 这里改成你的路径 ======
    tif_path = "/intelnvme04/jiang.mingyu/slum/GoogleMap/Lahore/2010/L19.tif"
    shp_path = "/intelnvme04/jiang.mingyu/slum/GoogleMap/Lahore/Lahore.shp"  # 模型输出多边形 shp，含 status 字段
    out_dir = "/intelnvme04/jiang.mingyu/slum/GoogleMap/Lahore/Lahore"  # 输出根目录（每张tif一个子目录）

    main(
        tif_path=tif_path,
        shp_path=shp_path,
        out_dir=out_dir,
        patch_size=1024,
        n_in=0,
        n_out=10,
        status_field="status",
        target_values=(0, 1),
    )

    # one_step()
