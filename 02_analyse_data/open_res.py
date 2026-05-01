import gzip
import struct
import numpy as np
import math
import os

# ----------------------------------------------------------------------
# Helper: read a vector (uint64 length + doubles)
# ----------------------------------------------------------------------
def read_vector(f):
    n_bytes = f.read(8)
    if len(n_bytes) < 8:
        raise ValueError("Unexpected EOF while reading vector length")
    (n,) = struct.unpack("<Q", n_bytes)
    if n > 1e9:
        raise ValueError(f"Unreasonable vector length: {n}")
    if n == 0:
        return np.array([], dtype=np.float64)
    data = f.read(n * 8)
    if len(data) < n * 8:
        raise ValueError("Unexpected EOF while reading vector data")
    return np.frombuffer(data, dtype=np.float64)

# ----------------------------------------------------------------------
# Unified reader that auto-detects old/new format
# ----------------------------------------------------------------------
def read_spde(fname):
    # Determine format by filename
    is_old = fname.endswith(".spde.gz")
    
    with gzip.open(fname, "rb") as f:
        # --- Magic header ---
        magic = f.read(8)
        if magic == b"SPDE001\x00":
            format_version = 1
        elif magic == b"SPDE002\x00":
            format_version = 2
        else:
            raise ValueError(f"Unknown magic header: {magic!r}")
        
        if format_version ==2:

            read = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt))) # type: ignore

            # --------------------------------------------------------------
            # Header
            # --------------------------------------------------------------
            header_version, = read("<I")
            model_id, = read("<I")

            # --------------------------------------------------------------
            # Parameter list (key-value pairs)
            # --------------------------------------------------------------
            n_params, = read("<I")

            params = {}
            for _ in range(n_params):
                name_bytes = f.read(32)
                name = name_bytes.rstrip(b"\x00").decode("utf-8") # type: ignore
                value, = read("<d")
                params[name] = value

            # store mode (from model_id)
            params["mode"] = model_id

            # --------------------------------------------------------------
            # Results metadata (UPDATED!)
            # --------------------------------------------------------------
            header_fmt = "<QQQQQQd"
            header_size = struct.calcsize(header_fmt)

            h_raw = f.read(header_size)
            if len(h_raw) < header_size:
                raise ValueError("Unexpected EOF while reading results header")

            (
                n_coarse,
                n_nodes,
                count_large,
                count_regular,
                count_overflow,
                count_invalid,
                runtime
            ) = struct.unpack(header_fmt, h_raw)   # type: ignore

            t_coarse   = read_vector(f)
            gamma_mat  = read_vector(f)
            nplus_mat  = read_vector(f)
            nminus_mat = read_vector(f) 
        else:
            # ==================================================================
            # OLD FORMAT
            # ==================================================================
            if is_old:
                params_fmt = "<i4x10dQ2d"   # old struct layout
                params_size = struct.calcsize(params_fmt)

                p_raw = f.read(params_size)
                if len(p_raw) < params_size:
                    raise ValueError("Unexpected EOF while reading old Params")

                vals = struct.unpack(params_fmt, p_raw) # type: ignore

                keys = [
                    "n", "fstar", "mu", "eta", "zeta", "beta",
                    "t_sub", "T", "dt", "mu_a", "Nmotor",
                    "seed", "ZETA_GAMMA_THRESHOLD", "LAMBDA_SMALL_THRESHOLD"
                ]
                params = dict(zip(keys, vals))

                # Old format does not store mode → infer it
                if params["Nmotor"] < 0:
                    params["mode"] = 0  # Deterministic
                else:
                    params["mode"] = 1  # Poisson (old format never used WhiteNoise)

            # ==================================================================
            # NEW FORMAT
            # ==================================================================
            else:
                # Must match your NEW save_compressed() writing order
                read = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt))) # type: ignore

                seed, = read("<Q")
                Nmotor, = read("<d")
                mu_a, = read("<d")
                n, = read("<i")
                fstar, = read("<d")
                mu, = read("<d")
                eta, = read("<d")
                zeta, = read("<d")
                beta, = read("<d")
                t_sub, = read("<d")
                T, = read("<d")
                dt, = read("<d")
                ZETA, = read("<d")
                LAMBDA, = read("<d")
                mode_int, = read("<i")

                params = dict(
                    seed=seed,
                    Nmotor=Nmotor,
                    mu_a=mu_a,
                    n=n,
                    fstar=fstar,
                    mu=mu,
                    eta=eta,
                    zeta=zeta,
                    beta=beta,
                    t_sub=t_sub,
                    T=T,
                    dt=dt,
                    ZETA_GAMMA_THRESHOLD=ZETA,
                    LAMBDA_SMALL_THRESHOLD=LAMBDA,
                    mode=mode_int
                )

            # --- Results header (same for both formats) ---
            header_fmt = "<QQQQd"
            h_raw = f.read(struct.calcsize(header_fmt))
            n_coarse, n_nodes, count_large, count_regular, runtime = struct.unpack(header_fmt, h_raw) # type: ignore

            # --- Vectors ---
            t_coarse   = read_vector(f)
            gamma_mat  = read_vector(f)
            nplus_mat  = read_vector(f)
            nminus_mat = read_vector(f)

    # --- Reshape matrices ---
    nt, ns = n_coarse, n_nodes

    is_3d = params.get("mode") in [7, 8]  # ThreeD_Deterministic / ThreeD_Poisson

    # --- gamma always 2D ---
    expected_gamma = nt * ns
    if gamma_mat.size != expected_gamma:
        raise ValueError(f"gamma_mat wrong size: {gamma_mat.size}, expected {expected_gamma}")

    gamma_mat = gamma_mat.reshape((nt, ns))

    # --- nplus / nminus ---
    if is_3d:
        expected = nt * 4 * ns
        if nplus_mat.size != expected:
            raise ValueError(f"nplus_mat wrong size: {nplus_mat.size}, expected {expected}")
        if nminus_mat.size != expected:
            raise ValueError(f"nminus_mat wrong size: {nminus_mat.size}, expected {expected}")

        # shape: (time, component, space)
        nplus_mat  = nplus_mat.reshape((nt, 4, ns))
        nminus_mat = nminus_mat.reshape((nt, 4, ns))

    else:
        expected = nt * ns
        if nplus_mat.size != expected:
            raise ValueError(f"nplus_mat wrong size: {nplus_mat.size}, expected {expected}")
        if nminus_mat.size != expected:
            raise ValueError(f"nminus_mat wrong size: {nminus_mat.size}, expected {expected}")

        nplus_mat  = nplus_mat.reshape((nt, ns))
        nminus_mat = nminus_mat.reshape((nt, ns))

    return {
        "params": params,
        "n_coarse": n_coarse,
        "n_nodes": n_nodes,
        "count_large_zeta_gamma": count_large,
        "count_regular_poisson": count_regular,
        "runtime_s": runtime,
        "t_coarse": t_coarse,
        "gamma_mat": gamma_mat,
        "nplus_mat": nplus_mat,
        "nminus_mat": nminus_mat,
    }

