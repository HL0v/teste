# %%
## ERA5


# %%
# -*- coding: utf-8 -*-
r"""
ERA5-Land monthly means -> tabela tidy por centróide (BR/US/AR), sem xarray (versão enxuta).

Mantém:
- Leitura de centróides com a MESMA lógica e caminhos do seu script original
- Download em blocos com .part + validação NetCDF/HDF5 + retries
- Leitura NetCDF com netCDF4.Dataset + num2date
- Amostragem nearest por centróide (lat/lon) e CSV "tidy"
- Preserva 'descricao' (AR) se existir

Mudanças:
- Código mais curto e direto; funções focadas
- Tratamento de 'expver' nas variáveis do NetCDF
- CLI Jupyter-safe (parse_known_args)
- Sem h5netcdf; apenas netCDF4

Saída padrão:
  C:\Users\joaog\Documents\Challenge_Itau_AlgoTrading_IA\data_raw\var_era5_land.csv
Colunas:
  country,admin_level,admin_code,admin_name,[descricao],date,t2m_c,tp_mm,u10,v10,ws10
"""

import os
import sys
import time
import argparse
import tem
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

# ==================== CREDENCIAIS (EMBUTIDAS) ====================
CDS_URL_HARDCODED = "https://cds.climate.copernicus.eu/api"
CDS_KEY_HARDCODED = "dad7839c-83a7-461b-99a0-a551517f8a24"  # cole sua chave aqui (se colar "UID:xxxxx" eu normalizo)

# ==================== PASTAS (MESMAS DO SEU SCRIPT) ====================
WORKDIR = Path(r"C:\Users\joaog\Documents\Challenge_Itau_AlgoTrading_IA\My_Code")
OUT_DIR  = WORKDIR.parent / "data_raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV  = OUT_DIR / "var_era5_land.csv"

ONEDRIVE_DOCS = Path(r"C:\Users\joaog\OneDrive\Documentos")

def first_existing(*candidates: Path) -> Path:
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    return Path(candidates[-1])

# Centrós padrão (iguais ao original)
DEF_BR = first_existing(
    ONEDRIVE_DOCS / "municipios_brasileiros_centroides.csv",
    WORKDIR        / "municipios_brasileiros_centroides.csv",
)
DEF_US = first_existing(ONEDRIVE_DOCS / "us_counties_centroides.csv")
DEF_AR = first_existing(ONEDRIVE_DOCS / "municipios_argentinos_centroides.csv")

DATA_DIR = WORKDIR.parent / "data"
DEF_CENT = DATA_DIR / "dim_admin_centroids.csv"

# Mapeamento de variáveis ERA5-Land
CDS_NAME = {
    "t2m": "2m_temperature",
    "tp":  "total_precipitation",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
}

TODAY_UTC_STR = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# ==================== HELPERS ====================
def _normalize_key(k: Optional[str]) -> str:
    k = (k or "").strip().strip('"').strip("'")
    return k.split(":", 1)[1] if ":" in k else k

def _is_valid_netcdf(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size < 1024:
            return False
        with open(path, "rb") as f:
            head = f.read(8)
        return head.startswith(b"CDF") or head == b"\x89HDF\r\n\x1a\n"
    except Exception:
        return False

def _make_cds_client():
    import cdsapi
    url = (CDS_URL_HARDCODED or os.getenv("CDSAPI_URL") or "https://cds.climate.copernicus.eu/api").strip()
    key = _normalize_key(CDS_KEY_HARDCODED) or _normalize_key(os.getenv("CDSAPI_KEY"))
    if key and key != "SUA_CHAVE_AQUI":
        return cdsapi.Client(url=url, key=key)
    return cdsapi.Client(url=url)  # .cdsapirc

def safe_retrieve(dataset: str, req: dict, out_nc: Path, retries: int = 2, sleep_base: float = 3.0) -> Path:
    out_nc = Path(out_nc)
    part = out_nc.with_suffix(out_nc.suffix + ".part")
    if part.exists():
        part.unlink()
    if out_nc.exists():
        out_nc.unlink()

    client = _make_cds_client()
    for attempt in range(retries + 1):
        print(f"[CDS] anos={req.get('year')} meses={req.get('month')} (tentativa {attempt+1}/{retries+1})…")
        client.retrieve(dataset, req, str(part))
        ok = _is_valid_netcdf(part)
        if ok:
            part.rename(out_nc)
            print(f"[CDS] OK → {out_nc} ({out_nc.stat().st_size:,} bytes)")
            return out_nc
        if attempt < retries:
            wait = sleep_base * (attempt + 1)
            print(f"[CDS] arquivo inválido; aguardando {wait:.1f}s…")
            time.sleep(wait)
    try:
        snippet = open(part, "rb").read(2000).decode("latin-1", "ignore")
        print("[CDS] Falha após re-tentativas. Prévia do arquivo baixado:\n", snippet)
    except Exception:
        print("[CDS] Falha após re-tentativas e não consegui ler .part")
    raise RuntimeError("Download não resultou em NetCDF válido.")

# ==================== IO DE CSV DE CENTRÓIDES (MESMO COMPORTAMENTO) ====================
def read_csv_smart(path: Path) -> pd.DataFrame:
    trials = [
        dict(sep=None, engine="python", encoding="utf-8",  decimal="."),
        dict(sep=None, engine="python", encoding="latin-1", decimal="."),
        dict(sep=";",  encoding="utf-8",  decimal=","),
        dict(sep=";",  encoding="latin-1", decimal=","),
        dict(sep=",",  encoding="utf-8",  decimal="."),
        dict(sep=",",  encoding="latin-1", decimal="."),
        dict(sep="\t", encoding="utf-8",  decimal="."),
        dict(sep="\t", encoding="latin-1", decimal="."),
    ]
    last_err = None
    for opts in trials:
        try:
            df = pd.read_csv(path, **opts)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as e:
            last_err = e
    raise ValueError(f"Não consegui ler {path} como CSV. Último erro: {last_err}")

def _pick_col(df, candidates, allow_contains=True):
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    low_map = {c.lower(): c for c in cols}
    for c in candidates:
        lc = str(c).lower()
        if lc in low_map:
            return low_map[lc]
    if allow_contains:
        for c in candidates:
            lc = str(c).lower()
            for col in cols:
                if lc in str(col).lower():
                    return col
    return None

def _standardize_centroids(df: pd.DataFrame,
                           country_code: str,
                           admin_level: str,
                           name_hints=None, code_hints=None, lat_hints=None, lon_hints=None) -> pd.DataFrame:
    name_hints = name_hints or [
        "admin_name","name","nome","nombre","municipio","município","municipio_nome",
        "name_muni","nome_municipio","Municipio","Município","NAME","Condado","County"
    ]
    code_hints = code_hints or [
        "Codigo_Municipio_IBGE","codigo_municipio_ibge",
        "GEOID","FIPS","fips","geoid",
        "admin_code","code","codigo","código","cd_mun","code_muni","cod_municipio",
        "codigo_ibge","cod_ibge","ibge","ibge_code","geocode","geocodigo","cd_geocmu",
        "codmun","id","Codigo_Condado","codigo_condado","Codigo_Municipio","municipio_id"
    ]
    lat_hints  = lat_hints  or ["lat","latitude","Latitude","centroide_lat"]
    lon_hints  = lon_hints  or ["lon","long","lng","longitude","Longitude","centroide_lon"]

    ncol = _pick_col(df, name_hints)
    ccol = _pick_col(df, code_hints)
    ycol = _pick_col(df, lat_hints)
    xcol = _pick_col(df, lon_hints)
    dcol = _pick_col(df, ["descricao","descrição","descripcion","descripción","description","desc"], allow_contains=True)

    if not all([ncol, ccol, ycol, xcol]):
        raise ValueError(f"[{country_code}] faltam colunas: name={ncol}, code={ccol}, lat={ycol}, lon={xcol}\nCols: {list(df.columns)}")

    out = pd.DataFrame({
        "country": country_code,
        "admin_level": admin_level,
        "admin_code": df[ccol].astype(str).str.strip(),
        "admin_name": df[ncol].astype(str).str.strip(),
        "lat": pd.to_numeric(df[ycol], errors="coerce"),
        "lon": pd.to_numeric(df[xcol], errors="coerce"),
    }).dropna(subset=["lat","lon"]).drop_duplicates(subset=["country","admin_code"])

    if dcol is not None:
        out["descricao"] = df[dcol].astype(str).str.strip()

    return out

def load_three_centroids(br_path: Path, us_path: Path, ar_path: Path) -> pd.DataFrame:
    frames = []

    # BR
    if br_path.exists():
        print(f"[leitura] BR ← {br_path}")
        br = read_csv_smart(br_path)
        if "uf" in br.columns and "municipio" in br.columns and "admin_name" not in br.columns:
            br["admin_name"] = br["municipio"].astype(str) + " - " + br["uf"].astype(str)
        frames.append(_standardize_centroids(
            br, "BR", "municipality",
            name_hints=["admin_name","Municipio","Município","municipio","name_muni","nome","name"],
            code_hints=[
                "Codigo_Municipio_IBGE","codigo_municipio_ibge","admin_code","cd_mun","code_muni",
                "cod_municipio","codigo","codigo_ibge","cod_ibge","ibge","ibge_code","geocode",
                "geocodigo","cd_geocmu","codmun","code"
            ],
            lat_hints=["lat","latitude","Latitude"],
            lon_hints=["lon","longitude","Longitude","long","lng"]
        ))
    else:
        print(f"[centroids] aviso: BR não encontrado em {br_path}")

    # US
    if us_path.exists():
        print(f"[leitura] US ← {us_path}")
        us = read_csv_smart(us_path)
        if "Codigo_Estado" in us.columns and "Codigo_Condado" in us.columns and "GEOID" not in us.columns:
            us["GEOID"] = (
                us["Codigo_Estado"].astype(str).str.zfill(2) +
                us["Codigo_Condado"].astype(str).str.zfill(3)
            )
        if "Condado" in us.columns and "admin_name" not in us.columns and "NAME" not in us.columns:
            us["admin_name"] = us["Condado"].astype(str)
        frames.append(_standardize_centroids(
            us, "US", "county",
            name_hints=["admin_name","Condado","County","NAME","name"],
            code_hints=["GEOID","FIPS","fips","geoid","Codigo_Condado","admin_code","code"],
            lat_hints=["lat","latitude","Latitude"],
            lon_hints=["lon","longitude","Longitude","long","lng"]
        ))
    else:
        print(f"[centroids] aviso: US não encontrado em {us_path}")

    # AR
    if ar_path.exists():
        print(f"[leitura] AR ← {ar_path}")
        ar = read_csv_smart(ar_path)
        mun_col  = _pick_col(ar, ["Municipio","municipio","nombre","name"])
        prov_col = _pick_col(ar, ["Provincia","provincia","provincia_nombre","prov_name","estado"])
        cat_col  = _pick_col(ar, ["categoria","category","tipo"])
        if mun_col and prov_col and "admin_name" not in ar.columns:
            ar["admin_name"] = ar[mun_col].astype(str) + " - " + ar[prov_col].astype(str)
        if mun_col and prov_col:
            base_desc = ar[mun_col].astype(str) + " - " + ar[prov_col].astype(str)
            ar["descricao"] = base_desc if not cat_col else base_desc + ", " + ar[cat_col].astype(str)

        frames.append(_standardize_centroids(
            ar, "AR", "municipio",
            name_hints=["admin_name","Municipio","municipio","nombre","name"],
            code_hints=["Codigo_Municipio","admin_code","id","municipio_id","code","codigo"],
            lat_hints=["lat","latitude","Latitude","centroide_lat"],
            lon_hints=["lon","longitude","Longitude","centroide_lon"]
        ))
    else:
        print(f"[centroids] aviso: AR não encontrado em {ar_path}")

    if not frames:
        raise FileNotFoundError(
            "Nenhum arquivo de centróides foi encontrado. "
            f"BR: {br_path}\nUS: {us_path}\nAR: {ar_path}"
        )
    df = pd.concat(frames, ignore_index=True)
    need = {"country","admin_level","admin_code","admin_name","lat","lon"}
    if not need.issubset(df.columns):
        raise ValueError("Padronização falhou. Esperado: country,admin_level,admin_code,admin_name,lat,lon")
    return df

# ==================== PERÍODO / BBOX ====================
def month_list_until(target_date: str) -> Tuple[List[str], int]:
    dt = datetime.strptime(target_date, "%Y-%m-%d")
    months = [f"{m:02d}" for m in range(1, dt.month + 1)]
    return months, dt.year

def get_bbox(df, margin_deg=1.0) -> List[float]:
    n = float(df["lat"].max() + margin_deg)
    s = float(df["lat"].min() - margin_deg)
    e = float(df["lon"].max() + margin_deg)
    w = float(df["lon"].min() - margin_deg)
    return [n, w, s, e]  # [N, W, S, E]

# ==================== LEITURA NETCDF ====================
def _find_coord_vars(nc):
    lat_name = "latitude" if "latitude" in nc.variables else ("lat" if "lat" in nc.variables else None)
    lon_name = "longitude" if "longitude" in nc.variables else ("lon" if "lon" in nc.variables else None)
    if not lat_name or not lon_name:
        raise ValueError("Variáveis de coordenadas não encontradas (latitude/longitude).")
    return lat_name, lon_name

def _points_to_grid_indices(lat_arr, lon_arr, pts_lonlat):
    lat_arr = np.asarray(lat_arr)
    lon_arr = np.asarray(lon_arr)
    lon_max = float(np.nanmax(lon_arr))
    adj_lon = np.mod(pts_lonlat["lon"].values, 360.0) if lon_max > 180.0 else pts_lonlat["lon"].values
    lat_idx = np.abs(lat_arr[None, :] - pts_lonlat["lat"].values[:, None]).argmin(axis=1)
    lon_idx = np.abs(lon_arr[None, :] - adj_lon[:, None]).argmin(axis=1)
    return lat_idx.astype(int), lon_idx.astype(int)

def _slice_time_to_2d(var, t_idx: int, lat_name: str, lon_name: str) -> np.ndarray:
    """
    Retorna arr 2D [lat,lon] para a variável no tempo t_idx.
    Tolera 'expver' (seleciona índice 0).
    """
    dims = list(var.dimensions)
    key = [slice(None)] * var.ndim
    if "time" in dims:
        key[dims.index("time")] = t_idx
    if "expver" in dims:
        key[dims.index("expver")] = 0
    arr = np.array(var[tuple(key)])
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Variável {getattr(var,'name','?')} não ficou 2D. dims={dims}, shape={arr.shape}")
    dims_no_time = [d for d in dims if d not in ("time","expver")]
    if len(dims_no_time) == 2 and dims_no_time == [lon_name, lat_name]:
        arr = arr.T
    return arr

def _read_block_to_tidy(nc_path: Path, pts: pd.DataFrame, short_vars: list) -> pd.DataFrame:
    from netCDF4 import Dataset, num2date
    nc = Dataset(str(nc_path), mode="r")
    try:
        lat_name, lon_name = _find_coord_vars(nc)
        lat_arr = nc.variables[lat_name][:]
        lon_arr = nc.variables[lon_name][:]
        tvar = nc.variables["time"]
        times = num2date(tvar[:], units=tvar.units, calendar=getattr(tvar, "calendar", "standard"))
        date_str = [f"{t.year:04d}-{t.month:02d}" for t in times]

        lat_idx, lon_idx = _points_to_grid_indices(lat_arr, lon_arr, pts)

        present: Dict[str,str] = {}
        for s in short_vars:
            full = CDS_NAME.get(s)
            if full in nc.variables:
                present[s] = full
        if not present:
            raise ValueError("Nenhuma das variáveis requisitadas está presente no NetCDF.")

        meta_cols = ["country","admin_level","admin_code","admin_name"]
        if "descricao" in pts.columns:
            meta_cols.append("descricao")
        meta = pts[meta_cols].copy().reset_index(drop=True)

        frames = []
        for ti, ds in enumerate(date_str):
            df_t = meta.copy(); df_t["date"] = ds
            for s, full in present.items():
                var = nc.variables[full]
                try: var.set_auto_maskandscale(True)
                except Exception: pass
                arr2d = _slice_time_to_2d(var, ti, lat_name, lon_name)
                vals = arr2d[lat_idx, lon_idx]
                if hasattr(vals, "mask"):
                    try: vals = vals.filled(np.nan)
                    except Exception: vals = np.asarray(vals, dtype=float)
                df_t[s] = np.asarray(vals, dtype=float)

            if "t2m" in df_t: df_t["t2m_c"] = df_t["t2m"] - 273.15
            if "tp"  in df_t: df_t["tp_mm"] = df_t["tp"] * 1000.0
            if "u10" in df_t and "v10" in df_t: df_t["ws10"] = np.sqrt(df_t["u10"]**2 + df_t["v10"]**2)

            keep = meta_cols + ["date"]
            for c in ("t2m_c","tp_mm","u10","v10","ws10"):
                if c in df_t.columns:
                    keep.append(c)
            frames.append(df_t[keep])

        return pd.concat(frames, ignore_index=True)
    finally:
        nc.close()

# ==================== RUNNER ====================
def run_era5(
    centroids: str = None,
    br_csv: str = str(DEF_BR),
    us_csv: str = str(DEF_US),
    ar_csv: str = str(DEF_AR),
    target_date: str = TODAY_UTC_STR,
    start_year: int = max(2001, datetime.now().year - 4),  # ~últimos 5 anos
    vars: str = "t2m,tp,u10,v10",
    block_years: int = 5,
    out_csv: str = str(OUT_CSV),
    retries: int = 2,
) -> pd.DataFrame:

    # 1) centróides (MESMA LÓGICA)
    if centroids and Path(centroids).exists():
        pts = read_csv_smart(Path(centroids))
        need = {"country","admin_level","admin_code","admin_name","lat","lon"}
        if not need.issubset(pts.columns):
            raise ValueError(f"CSV único precisa colunas: {sorted(need)}")
    else:
        pts = load_three_centroids(Path(br_csv), Path(us_csv), Path(ar_csv))

    # 2) variáveis CDS
    short = [v.strip() for v in vars.split(",") if v.strip()]
    cds_vars = [CDS_NAME[v] for v in short if v in CDS_NAME]
    if not cds_vars:
        raise ValueError("Use variáveis válidas (t2m,tp,u10,v10).")

    # 3) período + bbox
    months_final, last_year = month_list_until(target_date)
    if start_year > last_year:
        raise ValueError("start-year maior que ano da target-date.")
    area = get_bbox(pts, margin_deg=1.0)

    # 4) download por blocos + amostragem
    blocks = []
    dataset = "reanalysis-era5-land-monthly-means"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); y = int(start_year)
        while y <= last_year:
            y_end  = min(y + int(block_years) - 1, last_year)
            years  = list(range(y, y_end + 1))
            months = months_final if y_end == last_year else [f"{m:02d}" for m in range(1,13)]
            tmp_nc = td / f"era5land_{y}_{y_end}.nc"
            req = {
                "product_type": "monthly_averaged_reanalysis",
                "variable": cds_vars,
                "year": [str(yy) for yy in years],
                "month": months,
                "time": "00:00",
                "area": area,      # [N, W, S, E]
                "format": "netcdf",
            }
            print(f"[CDS] bloco {y}-{y_end}  meses={','.join(months)}")
            safe_retrieve(dataset, req, tmp_nc, retries=retries)

            blocks.append(_read_block_to_tidy(tmp_nc, pts, short_vars=short))
            y = y_end + 1

    tidy = (pd.concat(blocks, ignore_index=True)
            .sort_values(["country","admin_code","date"])
            .reset_index(drop=True))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out_csv, index=False)
    print(f"OK: {out_csv}  linhas={len(tidy):,}")
    return tidy

# ==================== CLI ====================
def parse_args(argv=None):
    p = argparse.ArgumentParser(add_help=True, description="ERA5-Land monthly means → CSV tidy por centróide")
    # centróides (iguais ao original: defaults apontando pros seus arquivos)
    p.add_argument("--br-csv", default=str(DEF_BR))
    p.add_argument("--us-csv", default=str(DEF_US))
    p.add_argument("--ar-csv", default=str(DEF_AR))
    p.add_argument("--centroids", default=str(DEF_CENT))

    # período/variáveis
    p.add_argument("--target-date", default=TODAY_UTC_STR)
    p.add_argument("--start-year", type=int, default=max(2001, datetime.now().year - 4))  # ~5 anos
    p.add_argument("--vars", default="t2m,tp,u10,v10")
    p.add_argument("--block-years", type=int, default=5)

    # saída/retries
    p.add_argument("--out-csv", default=str(OUT_CSV))
    p.add_argument("--retries", type=int, default=2)
    return p.parse_known_args(argv)[0]  # Jupyter-safe (ignora --f=...)

def main(argv=None):
    a = parse_args(argv)
    run_era5(
        centroids=a.centroids if Path(a.centroids).exists() else None,
        br_csv=a.br_csv, us_csv=a.us_csv, ar_csv=a.ar_csv,
        target_date=a.target_date, start_year=a.start_year,
        vars=a.vars, block_years=a.block_years, out_csv=a.out_csv,
        retries=a.retries,
    )

if __name__ == "__main__":
    main(sys.argv[1:])


