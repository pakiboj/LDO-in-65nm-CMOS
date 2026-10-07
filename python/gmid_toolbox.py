
"""
gm/ID LUT Toolbox
=================

Standalone toolbox za MOS gm/ID LUT baze u MATLAB .mat formatu.

Očekivani fajlovi po default-u:
    65nch.mat  -> MATLAB promenljiva "nch"
    65pch.mat  -> MATLAB promenljiva "pch"

Glavni API:
    tb.LUT1(tranz, gmId, VDS, VSB, L_nm) -> VGS / VSG
    tb.LUT2(tranz, gmId, VGS, VSB, L_nm) -> VDS / VSD
    tb.LUT3(tranz, VGS, VDS, VSB, L_nm)  -> gm/ID [1/V]
    tb.LUT4(tranz, VGS, VDS, VSB, L_nm)  -> gds/ID [1/V]

Kompatibilni opisni API:
    tb.vgs(...)              -> VGS za NMOS / VSG za PMOS
    tb.vds(...)              -> VDS za NMOS / VSD za PMOS
    tb.gm_id_at_bias(...)    -> gm/ID za terminalne napone
    tb.gds_id_at_bias(...)   -> gds/ID za terminalne napone
    tb.gds_id(...)           -> gds/ID preko zadatog gm/ID
    tb.jd(...)               -> JD = ID/W [A/um]
    tb.capacitances_at_bias(...) -> kapacitivnosti iz LUT baze [F]

Za PMOS se svi naponi zadaju kao POZITIVNE magnitude:
    VSG, VSD, VBS

Primer:
    from gmid_toolbox import GMIDToolbox

    tb = GMIDToolbox()

    VSG = tb.vgs(
        mos="pmos",
        gm_id=8,
        vds=0.2,      # zapravo VSD
        L_nm=65,
        vsb=0.0,      # zapravo VBS
    )
"""

from pathlib import Path
from itertools import permutations

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq


class LUTError(RuntimeError):
    """Greška kada tražena radna tačka nije dostupna u karakterisanoj LUT bazi."""
    pass


def _mat_to_dict(obj):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "_fieldnames"):
        return {name: getattr(obj, name) for name in obj._fieldnames}
    raise TypeError(f"Nepodržan MATLAB objekat: {type(obj)}")


def _pick_field(d, name):
    for key, value in d.items():
        if str(key).upper() == name.upper():
            return value
    raise KeyError(
        f"Polje '{name}' nije pronađeno. Dostupna polja: {list(d.keys())}"
    )


def _to_um_if_needed(x):
    x = np.asarray(x, dtype=float)
    finite = np.abs(x[np.isfinite(x)])
    if finite.size and finite.max() < 1e-3:
        return x * 1e6
    return x


class _MOSDatabase:
    TARGET_ORDER = ("L", "VGS", "VDS", "VSB")

    def __init__(
        self,
        mat_path,
        variable,
        kind,
        axis_order_hint=None,
    ):
        self.path = Path(mat_path)
        self.variable = variable
        self.kind = kind.lower()
        self.axis_order_hint = axis_order_hint

        if self.kind not in ("nmos", "pmos"):
            raise ValueError("kind mora biti 'nmos' ili 'pmos'.")

        if not self.path.exists():
            raise FileNotFoundError(f"Nedostaje LUT fajl: {self.path}")

        raw = loadmat(
            self.path,
            squeeze_me=True,
            struct_as_record=False,
        )

        if variable not in raw:
            candidates = [k for k in raw if not k.startswith("__")]
            raise KeyError(
                f"Promenljiva '{variable}' nije pronađena u {self.path}. "
                f"Dostupno: {candidates}"
            )

        self.raw = _mat_to_dict(raw[variable])

        L_raw = np.atleast_1d(np.squeeze(_pick_field(self.raw, "L"))).astype(float)
        VGS_raw = np.atleast_1d(np.squeeze(_pick_field(self.raw, "VGS"))).astype(float)
        VDS_raw = np.atleast_1d(np.squeeze(_pick_field(self.raw, "VDS"))).astype(float)

        try:
            VSB_raw = np.atleast_1d(
                np.squeeze(_pick_field(self.raw, "VSB"))
            ).astype(float)
        except KeyError:
            VSB_raw = np.array([0.0])

        self._raw_axis_lengths = {
            "L": len(L_raw),
            "VGS": len(VGS_raw),
            "VDS": len(VDS_raw),
            "VSB": len(VSB_raw),
        }

        L_pub = _to_um_if_needed(L_raw)

        # Javni PMOS API koristi pozitivne magnitude VSG/VSD/VBS.
        if self.kind == "pmos":
            VGS_pub = np.abs(VGS_raw)
            VDS_pub = np.abs(VDS_raw)
            VSB_pub = np.abs(VSB_raw)
        else:
            VGS_pub = VGS_raw.copy()
            VDS_pub = VDS_raw.copy()
            VSB_pub = VSB_raw.copy()

        self.data = {}

        for field in (
            "ID", "GM", "GDS", "W",
            "CGG", "CGS", "CSG", "CGD", "CDG", "CGB", "CDD", "CSS",
        ):
            arr = np.asarray(_pick_field(self.raw, field), dtype=float)
            arr = self._coerce_to_target_order(arr)

            if field == "W":
                arr = _to_um_if_needed(np.abs(arr))
            else:
                # ID/GM/GDS i kapacitivnosti ostaju u SI jedinicama iz LUT baze.
                arr = np.abs(arr)

            self.data[field] = arr

        axes = {
            "L": np.asarray(L_pub, dtype=float),
            "VGS": np.asarray(VGS_pub, dtype=float),
            "VDS": np.asarray(VDS_pub, dtype=float),
            "VSB": np.asarray(VSB_pub, dtype=float),
        }

        # Posle abs() za PMOS sortiramo ose zajedno sa podacima.
        for axis_index, axis_name in enumerate(self.TARGET_ORDER):
            vals = axes[axis_name]
            order = np.argsort(vals)
            vals = vals[order]

            if np.any(np.diff(vals) <= 0):
                raise LUTError(
                    f"Osa {axis_name} nije strogo rastuća nakon normalizacije: {vals}"
                )

            axes[axis_name] = vals
            for field in self.data:
                self.data[field] = np.take(
                    self.data[field],
                    order,
                    axis=axis_index,
                )

        self.L = axes["L"]
        self.VGS = axes["VGS"]
        self.VDS = axes["VDS"]
        self.VSB = axes["VSB"]

        self.grid = (self.L, self.VGS, self.VDS, self.VSB)

        self.interp = {
            name: RegularGridInterpolator(
                self.grid,
                arr,
                bounds_error=False,
                fill_value=np.nan,
                method="linear",
            )
            for name, arr in self.data.items()
        }

        self._curve_cache = {}

    def _coerce_to_target_order(self, arr):
        arr = np.squeeze(np.asarray(arr, dtype=float))

        if arr.ndim == 0:
            shape = tuple(
                self._raw_axis_lengths[a]
                for a in self.TARGET_ORDER
            )
            return np.full(shape, float(arr))

        active = [
            a for a in self.TARGET_ORDER
            if self._raw_axis_lengths[a] > 1
        ]

        if arr.ndim != len(active):
            raise LUTError(
                f"Shape {arr.shape} nije kompatibilan sa aktivnim osama {active}."
            )

        if self.axis_order_hint is not None:
            source_order = tuple(
                a for a in self.axis_order_hint
                if self._raw_axis_lengths[a] > 1
            )
            expected = tuple(
                self._raw_axis_lengths[a]
                for a in source_order
            )
            if expected != arr.shape:
                raise LUTError(
                    f"AXIS_ORDER_HINT očekuje shape {expected}, "
                    f"a polje ima {arr.shape}."
                )
        else:
            candidates = []
            for p in permutations(active):
                shape = tuple(
                    self._raw_axis_lengths[a]
                    for a in p
                )
                if shape == arr.shape:
                    candidates.append(p)

            if not candidates:
                raise LUTError(
                    f"Ne mogu da odredim redosled osa za shape {arr.shape}."
                )

            preferred = [
                tuple(a for a in ("L", "VGS", "VDS", "VSB") if a in active),
                tuple(a for a in ("L", "VSB", "VGS", "VDS") if a in active),
                tuple(a for a in ("VSB", "VGS", "VDS", "L") if a in active),
                tuple(a for a in ("VGS", "VDS", "VSB", "L") if a in active),
            ]

            source_order = next(
                (p for p in preferred if p in candidates),
                candidates[0],
            )

        permutation = [
            source_order.index(a)
            for a in active
        ]
        arr = np.transpose(arr, axes=permutation)

        for i, axis_name in enumerate(self.TARGET_ORDER):
            if self._raw_axis_lengths[axis_name] == 1:
                arr = np.expand_dims(arr, axis=i)

        target_shape = tuple(
            self._raw_axis_lengths[a]
            for a in self.TARGET_ORDER
        )

        if arr.shape != target_shape:
            raise LUTError(
                f"Interna greška: dobijen shape {arr.shape}, "
                f"očekivan {target_shape}."
            )

        return arr

    def _bias_values_many(self, *, L_um, vgs, vds, vsb=0.0):
        vgs = np.asarray(vgs, dtype=float)
        vds = np.asarray(vds, dtype=float)
        vsb = np.asarray(vsb, dtype=float)

        vgs, vds, vsb = np.broadcast_arrays(vgs, vds, vsb)
        shape = vgs.shape

        pts = np.column_stack([
            np.full(vgs.size, float(L_um)),
            vgs.ravel(),
            vds.ravel(),
            vsb.ravel(),
        ])

        ID = self.interp["ID"](pts)
        GM = self.interp["GM"](pts)
        GDS = self.interp["GDS"](pts)
        W = self.interp["W"](pts)
        capacitances = {
            name: self.interp[name](pts)
            for name in ("CGG", "CGS", "CSG", "CGD", "CDG", "CGB", "CDD", "CSS")
        }

        with np.errstate(divide="ignore", invalid="ignore"):
            gm_id = GM / ID
            gds_id = GDS / ID
            jd = ID / W

        result = {
            "ID": ID.reshape(shape),
            "GM": GM.reshape(shape),
            "GDS": GDS.reshape(shape),
            "W": W.reshape(shape),
            "gm_id": gm_id.reshape(shape),
            "gds_id": gds_id.reshape(shape),
            "jd": jd.reshape(shape),
        }
        result.update({name: value.reshape(shape) for name, value in capacitances.items()})
        return result

    def _bias_values(self, *, L_um, vgs, vds, vsb=0.0):
        vals = self._bias_values_many(
            L_um=L_um,
            vgs=np.array([vgs]),
            vds=np.array([vds]),
            vsb=np.array([vsb]),
        )

        out = {
            k: float(np.ravel(v)[0])
            for k, v in vals.items()
        }

        if not np.all(np.isfinite([
            out["ID"], out["GM"], out["GDS"], out["W"]
        ])):
            raise LUTError(
                f"{self.kind}: radna tačka je van LUT opsega: "
                f"L={L_um} um, "
                f"VGS/VSG={vgs} V, "
                f"VDS/VSD={vds} V, "
                f"VSB/VBS={vsb} V."
            )

        return out

    def _curve_vs_vgs(self, *, L_um, vds, vsb=0.0):
        key = (
            round(float(L_um), 12),
            round(float(vds), 10),
            round(float(vsb), 10),
        )

        if key in self._curve_cache:
            return self._curve_cache[key]

        vgs_axis = self.VGS.copy()

        vals = self._bias_values_many(
            L_um=L_um,
            vgs=vgs_axis,
            vds=np.full_like(vgs_axis, float(vds)),
            vsb=np.full_like(vgs_axis, float(vsb)),
        )

        gm_id = vals["gm_id"]
        gds_id = vals["gds_id"]
        jd = vals["jd"]

        valid = (
            np.isfinite(gm_id)
            & np.isfinite(gds_id)
            & np.isfinite(jd)
            & (jd > 0)
        )

        result = (
            vgs_axis[valid],
            gm_id[valid],
            gds_id[valid],
            jd[valid],
        )

        self._curve_cache[key] = result
        return result

    @staticmethod
    def _interp_on_gmid(gm_id_target, gm_id_axis, y_axis):
        gm_id_axis = np.asarray(gm_id_axis, dtype=float)
        y_axis = np.asarray(y_axis, dtype=float)

        valid = np.isfinite(gm_id_axis) & np.isfinite(y_axis)
        x = gm_id_axis[valid]
        y = y_axis[valid]

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        x_unique, idx = np.unique(
            np.round(x, 12),
            return_index=True,
        )
        y_unique = y[idx]

        if len(x_unique) < 2:
            raise LUTError("Premalo validnih gm/ID tačaka za interpolaciju.")

        if gm_id_target < x_unique[0] or gm_id_target > x_unique[-1]:
            raise LUTError(
                f"gm/ID={gm_id_target:.4g} 1/V je van dostupnog lokalnog opsega "
                f"[{x_unique[0]:.4g}, {x_unique[-1]:.4g}] 1/V."
            )

        return float(
            np.interp(
                gm_id_target,
                x_unique,
                y_unique,
            )
        )

    def vgs(self, *, gm_id, vds, L_um, vsb=0.0):
        """
        Vraća VGS [V] za NMOS ili pozitivni VSG [V] za PMOS.
        """
        vgs_axis, gmids, _, _ = self._curve_vs_vgs(
            L_um=L_um,
            vds=vds,
            vsb=vsb,
        )
        target = float(gm_id)

        # Prvo proveravamo opseg i dobijamo dobru rezervnu aproksimaciju.
        estimate = self._interp_on_gmid(target, gmids, vgs_axis)
        delta = np.asarray(gmids, dtype=float) - target
        roots = []

        exact = np.flatnonzero(np.isclose(delta, 0.0, rtol=1e-11, atol=1e-13))
        roots.extend(float(vgs_axis[index]) for index in exact)

        finite_pairs = np.isfinite(delta[:-1]) & np.isfinite(delta[1:])
        crossings = np.flatnonzero(
            finite_pairs & (delta[:-1] * delta[1:] < 0)
        )
        for i in crossings:
            x0 = float(vgs_axis[i])
            x1 = float(vgs_axis[i + 1])
            op0 = self._bias_values(L_um=L_um, vgs=x0, vds=vds, vsb=vsb)
            op1 = self._bias_values(L_um=L_um, vgs=x1, vds=vds, vsb=vsb)

            # RegularGridInterpolator je linearan u VGS za ID i GM.
            # Jednačina (GM0+t*dGM)/(ID0+t*dID)=target zato ima
            # zatvoreno rešenje i ne zahteva iterativni root solver.
            d_id = op1["ID"] - op0["ID"]
            d_gm = op1["GM"] - op0["GM"]
            denominator = target * d_id - d_gm

            if abs(denominator) > 1e-30:
                t = (op0["GM"] - target * op0["ID"]) / denominator
                if -1e-12 <= t <= 1.0 + 1e-12:
                    roots.append(x0 + float(np.clip(t, 0.0, 1.0)) * (x1 - x0))
                    continue

            def root_fun(vgs_local):
                return self.gm_id_at_bias(
                    vgs=vgs_local,
                    vds=vds,
                    L_um=L_um,
                    vsb=vsb,
                ) - target

            roots.append(float(brentq(root_fun, x0, x1, maxiter=100)))

        if roots:
            # gm/ID(VGS) je u normalnom opsegu monotona funkcija. Ako baza
            # ipak daje više preseka, zadržavamo onaj blizak stabilnoj
            # interpolacionoj proceni.
            return min(roots, key=lambda value: abs(value - estimate))

        return estimate

    def gm_id_at_bias(self, *, vgs, vds, L_um, vsb=0.0):
        """Vraća direktni gm/ID [1/V] pri zadatim terminalnim naponima."""
        return self._bias_values(
            L_um=L_um,
            vgs=vgs,
            vds=vds,
            vsb=vsb,
        )["gm_id"]

    def gds_id_at_bias(self, *, vgs, vds, L_um, vsb=0.0):
        """Vraća direktni gds/ID [1/V] pri zadatim terminalnim naponima."""
        return self._bias_values(
            L_um=L_um,
            vgs=vgs,
            vds=vds,
            vsb=vsb,
        )["gds_id"]

    def jd_at_bias(self, *, vgs, vds, L_um, vsb=0.0):
        """Vraća direktni JD=ID/W [A/um] pri zadatim terminalnim naponima."""
        return self._bias_values(
            L_um=L_um,
            vgs=vgs,
            vds=vds,
            vsb=vsb,
        )["jd"]

    def capacitances_at_bias(self, *, vgs, vds, L_um, vsb=0.0):
        """Vraća LUT kapacitivnosti i referentnu širinu pri radnoj tački."""
        values = self._bias_values(
            L_um=L_um, vgs=vgs, vds=vds, vsb=vsb,
        )
        names = ("CGG", "CGS", "CSG", "CGD", "CDG", "CGB", "CDD", "CSS")
        result = {name: values[name] for name in names}
        result["W_REF_UM"] = values["W"]
        return result

    def gds_id(self, *, gm_id, vds, L_um, vsb=0.0):
        """
        Vraća gds/ID [1/V].
        """
        _, gmids, gdsids, _ = self._curve_vs_vgs(
            L_um=L_um,
            vds=vds,
            vsb=vsb,
        )
        return self._interp_on_gmid(
            gm_id,
            gmids,
            gdsids,
        )

    def jd(self, *, gm_id, vds, L_um, vsb=0.0):
        """
        Vraća JD = ID/W [A/um].
        """
        _, gmids, _, jds = self._curve_vs_vgs(
            L_um=L_um,
            vds=vds,
            vsb=vsb,
        )
        return self._interp_on_gmid(
            gm_id,
            gmids,
            jds,
        )

    def vds(
        self,
        *,
        gm_id,
        vgs,
        L_um,
        vsb=0.0,
        n_scan=301,
        relative_tolerance=1e-3,
        preferred_vds=None,
    ):
        """
        Vraća VDS [V] za NMOS ili pozitivni VSD [V] za PMOS.

        Rešava:
            gm/ID(VGS, VDS, L, VSB) = zadati gm/ID

        Ako postoji više preseka, podrazumevano vraća najmanji validni
        VDS/VSD. Kada je zadat ``preferred_vds``, vraća presek koji je
        najbliži toj vrednosti; ovo je važno za praćenje iste grane rešenja
        u parametarskom sweep-u.
        """
        if preferred_vds is not None:
            preferred_vds = float(preferred_vds)
            try:
                gm_preferred = self.gm_id_at_bias(
                    vgs=vgs,
                    vds=preferred_vds,
                    L_um=L_um,
                    vsb=vsb,
                )
                preferred_error = abs(gm_preferred - float(gm_id)) / max(
                    abs(float(gm_id)), 1e-12
                )
                if preferred_error <= relative_tolerance:
                    return preferred_vds
            except LUTError:
                pass

        vds_axis = np.linspace(
            float(self.VDS[0]),
            float(self.VDS[-1]),
            int(n_scan),
        )

        vals = self._bias_values_many(
            L_um=L_um,
            vgs=np.full_like(vds_axis, float(vgs)),
            vds=vds_axis,
            vsb=np.full_like(vds_axis, float(vsb)),
        )

        gm_actual = vals["gm_id"]
        f = gm_actual - float(gm_id)

        valid = np.isfinite(gm_actual) & np.isfinite(f)

        x = vds_axis[valid]
        f = f[valid]
        gm_valid = gm_actual[valid]

        if len(x) < 2:
            raise LUTError("Nema dovoljno validnih VDS/VSD tačaka za inverse LUT.")

        # Kandidat iz skenirane mreže. Ne vraćamo ga odmah, jer može
        # postojati više preseka i korisnik može tražiti određenu granu.
        k = int(np.argmin(np.abs(f)))
        rel_err = abs(f[k]) / max(abs(float(gm_id)), 1e-12)

        # Svi VDS/VSD preseci.
        roots = []

        for i in range(len(x) - 1):
            if f[i] == 0:
                roots.append(float(x[i]))
                continue

            if f[i] * f[i + 1] < 0:

                def root_fun(vds_local):
                    vals_local = self._bias_values(
                        L_um=L_um,
                        vgs=vgs,
                        vds=vds_local,
                        vsb=vsb,
                    )
                    return vals_local["gm_id"] - gm_id

                root = brentq(
                    root_fun,
                    x[i],
                    x[i + 1],
                    maxiter=100,
                )
                roots.append(float(root))

        if not roots and rel_err <= relative_tolerance:
            roots.append(float(x[k]))

        if roots:
            roots = np.asarray(sorted(set(np.round(roots, 12))), dtype=float)
            if preferred_vds is not None:
                preferred_vds = float(preferred_vds)
                return float(roots[np.argmin(np.abs(roots - preferred_vds))])
            return float(roots[0])

        raise LUTError(
            f"Nije pronađen VDS/VSD za gm/ID={gm_id:g} 1/V i "
            f"VGS/VSG={vgs:g} V. U dostupnom VDS/VSD opsegu "
            f"gm/ID je približno [{np.nanmin(gm_valid):.4g}, "
            f"{np.nanmax(gm_valid):.4g}] 1/V."
        )

    def info(self):
        return {
            "kind": self.kind,
            "L_um": (float(self.L[0]), float(self.L[-1])),
            "VGS_or_VSG_V": (float(self.VGS[0]), float(self.VGS[-1])),
            "VDS_or_VSD_V": (float(self.VDS[0]), float(self.VDS[-1])),
            "VSB_or_VBS_V": (float(self.VSB[0]), float(self.VSB[-1])),
        }


class GMIDToolbox:
    """
    Javni gm/ID LUT API.

    Sve dužine se zadaju u nm preko parametra L_nm.

    Za PMOS:
        vgs -> VSG (pozitivna magnituda)
        vds -> VSD (pozitivna magnituda)
        vsb -> VBS (pozitivna magnituda)
    """

    def __init__(
        self,
        nmos_mat="65nch.mat",
        pmos_mat="65pch.mat",
        nmos_variable="nch",
        pmos_variable="pch",
        axis_order_hint=None,
    ):
        self.nmos = _MOSDatabase(
            nmos_mat,
            nmos_variable,
            "nmos",
            axis_order_hint=axis_order_hint,
        )

        self.pmos = _MOSDatabase(
            pmos_mat,
            pmos_variable,
            "pmos",
            axis_order_hint=axis_order_hint,
        )

    def _db(self, mos):
        mos = str(mos).lower()

        if mos in ("nmos", "nch", "n"):
            return self.nmos

        if mos in ("pmos", "pch", "p"):
            return self.pmos

        raise ValueError(
            "tranz/mos mora biti 'nch'/'nmos' ili 'pch'/'pmos'."
        )

    @staticmethod
    def _L_um(L_nm):
        return float(L_nm) / 1000.0

    def vgs(self, *, mos, gm_id, vds, L_nm, vsb=0.0):
        """
        LUT za VGS/VSG.

        Parameters
        ----------
        mos : "nmos" ili "pmos"
        gm_id : [1/V]
        vds : NMOS VDS [V] / PMOS VSD [V]
        L_nm : dužina [nm]
        vsb : NMOS VSB [V] / PMOS VBS [V]

        Returns
        -------
        NMOS VGS [V] / PMOS VSG [V]
        """
        return self._db(mos).vgs(
            gm_id=float(gm_id),
            vds=float(vds),
            L_um=self._L_um(L_nm),
            vsb=float(vsb),
        )

    def vds(
        self,
        *,
        mos,
        gm_id,
        vgs,
        L_nm,
        vsb=0.0,
        n_scan=301,
        relative_tolerance=1e-3,
        preferred_vds=None,
    ):
        """
        Inverzni LUT za VDS/VSD.

        Parameters
        ----------
        mos : "nmos" ili "pmos"
        gm_id : [1/V]
        vgs : NMOS VGS [V] / PMOS VSG [V]
        L_nm : dužina [nm]
        vsb : NMOS VSB [V] / PMOS VBS [V]

        Returns
        -------
        NMOS VDS [V] / PMOS VSD [V]
        """
        return self._db(mos).vds(
            gm_id=float(gm_id),
            vgs=float(vgs),
            L_um=self._L_um(L_nm),
            vsb=float(vsb),
            n_scan=int(n_scan),
            relative_tolerance=float(relative_tolerance),
            preferred_vds=(
                None if preferred_vds is None else float(preferred_vds)
            ),
        )

    def gm_id_at_bias(self, *, mos, vgs, vds, L_nm, vsb=0.0):
        """Direktni gm/ID [1/V] za zadate terminalne napone."""
        return self._db(mos).gm_id_at_bias(
            vgs=float(vgs),
            vds=float(vds),
            L_um=self._L_um(L_nm),
            vsb=float(vsb),
        )

    def gds_id_at_bias(self, *, mos, vgs, vds, L_nm, vsb=0.0):
        """Direktni gds/ID [1/V] za zadate terminalne napone."""
        return self._db(mos).gds_id_at_bias(
            vgs=float(vgs),
            vds=float(vds),
            L_um=self._L_um(L_nm),
            vsb=float(vsb),
        )

    def jd_at_bias(self, *, mos, vgs, vds, L_nm, vsb=0.0):
        """Direktni JD=ID/W [A/um] za zadate terminalne napone."""
        return self._db(mos).jd_at_bias(
            vgs=float(vgs),
            vds=float(vds),
            L_um=self._L_um(L_nm),
            vsb=float(vsb),
        )

    def capacitances_at_bias(self, *, mos, vgs, vds, L_nm, vsb=0.0):
        """LUT kapacitivnosti [F] pri zadatim terminalnim naponima.

        Vrednosti odgovaraju referentnoj širini ``W_REF_UM`` iz LUT baze.
        Za drugu ukupnu širinu skaliraju se faktorom W/W_REF_UM.
        """
        return self._db(mos).capacitances_at_bias(
            vgs=float(vgs), vds=float(vds),
            L_um=self._L_um(L_nm), vsb=float(vsb),
        )

    def gds_id(self, *, mos, gm_id, vds, L_nm, vsb=0.0):
        """
        LUT za gds/ID [1/V].
        """
        return self._db(mos).gds_id(
            gm_id=float(gm_id),
            vds=float(vds),
            L_um=self._L_um(L_nm),
            vsb=float(vsb),
        )

    def jd(self, *, mos, gm_id, vds, L_nm, vsb=0.0):
        """
        LUT za JD = ID/W [A/um].
        """
        return self._db(mos).jd(
            gm_id=float(gm_id),
            vds=float(vds),
            L_um=self._L_um(L_nm),
            vsb=float(vsb),
        )

    # Aliasi sa LUT imenima, ako ti je čitljivije u projektnom kodu:
    LUT_VGS = vgs
    LUT_VDS = vds
    LUT_GDS_ID = gds_id
    LUT_JD = jd

    # ----------------------- LUT1 ... LUT4 API -----------------------
    # Redosled argumenata prati matematički zapis iz LDO projekta.
    def LUT1(self, tranz, gm_id, vds, vsb, L_nm):
        """VGS/VSG = LUT1(tranz, gm/ID, VDS/VSD, VSB/VBS, L_nm)."""
        return self.vgs(
            mos=tranz,
            gm_id=gm_id,
            vds=vds,
            vsb=vsb,
            L_nm=L_nm,
        )

    def LUT2(
        self,
        tranz,
        gm_id,
        vgs,
        vsb,
        L_nm,
        *,
        preferred_vds=None,
        n_scan=301,
        relative_tolerance=1e-3,
    ):
        """VDS/VSD = LUT2(tranz, gm/ID, VGS/VSG, VSB/VBS, L_nm).

        Rezultat je terminalna razlika napona, a ne apsolutni napon
        drain čvora. Za NMOS sa source naponom VS važi VD=VS+VDS.
        """
        return self.vds(
            mos=tranz,
            gm_id=gm_id,
            vgs=vgs,
            vsb=vsb,
            L_nm=L_nm,
            preferred_vds=preferred_vds,
            n_scan=n_scan,
            relative_tolerance=relative_tolerance,
        )

    def LUT3(self, tranz, vgs, vds, vsb, L_nm):
        """gm/ID = LUT3(tranz, VGS/VSG, VDS/VSD, VSB/VBS, L_nm)."""
        return self.gm_id_at_bias(
            mos=tranz,
            vgs=vgs,
            vds=vds,
            vsb=vsb,
            L_nm=L_nm,
        )

    def LUT4(self, tranz, vgs, vds, vsb, L_nm):
        """gds/ID = LUT4(tranz, VGS/VSG, VDS/VSD, VSB/VBS, L_nm)."""
        return self.gds_id_at_bias(
            mos=tranz,
            vgs=vgs,
            vds=vds,
            vsb=vsb,
            L_nm=L_nm,
        )

    # Mala slova su praktična u Python kodu, a velika prate zadatak.
    lut1 = LUT1
    lut2 = LUT2
    lut3 = LUT3
    lut4 = LUT4

    def info(self):
        return {
            "nmos": self.nmos.info(),
            "pmos": self.pmos.info(),
        }
