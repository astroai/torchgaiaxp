"""ESA Gaia Archive TAP/ADQL and DataLink HTTPS client (stdlib only, no astroquery required)."""

from __future__ import annotations

import http.cookiejar
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from .io import XPBatch, read_fits, read_parquet, read_votable, read_xp_continuous, read_zip

DEFAULT_ARCHIVE_SERVER = "https://gea.esac.esa.int/"


def validate_source_ids(source_ids) -> list[int]:
    """Validate and normalize an iterable of Gaia ``source_id`` integers."""
    if isinstance(source_ids, (str, bytes)):
        raise TypeError("source_ids must be an iterable of integer source IDs, not a string")
    out: list[int] = []
    for raw in source_ids:
        if isinstance(raw, (bool, np.bool_)):
            raise TypeError(f"invalid source_id {raw!r}: booleans are not valid source IDs")
        if isinstance(raw, (float, np.floating)):
            raise TypeError(f"invalid source_id {raw!r}: floats are not valid source IDs")
        if isinstance(raw, (int, np.integer)):
            out.append(int(raw))
        elif isinstance(raw, str):
            s = raw.strip()
            if not s or (s[0] == "-" and not s[1:].isdigit()) or (s[0] != "-" and not s.isdigit()):
                raise ValueError(f"invalid source_id string {raw!r}")
            out.append(int(s))
        else:
            raise TypeError(f"invalid source_id type {type(raw).__name__}")
    if not out:
        raise ValueError("source_ids must not be empty")
    return out


def _read_credentials_file(path: str | Path) -> tuple[str, str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    non_empty = [line.strip() for line in lines if line.strip()]
    if len(non_empty) < 2:
        raise ValueError(f"credentials_file {path!r} must contain username and password lines")
    return non_empty[0], non_empty[1]


def _archive_release_label(release: str | None) -> str:
    rel = "dr3" if release is None else str(release).lower()
    return f"Gaia {rel.upper()}"


def _decode_archive_payload(raw: bytes, content_type: str = "") -> XPBatch:
    ct = content_type.lower()
    if raw[:2] == b"PK" or "zip" in ct:
        return read_zip(raw)
    if raw[:4] == b"PAR1" or "parquet" in ct:
        return read_parquet(raw)
    head = raw[:512].lstrip()
    if head.startswith(b"SIMPLE  =") or "fits" in ct:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".fits") as tmp:
            tmp.write(raw)
            tmp.flush()
            return read_fits(tmp.name)
    if head.startswith(b"<?xml") or b"<VOTABLE" in head or "votable" in ct or "xml" in ct:
        return read_votable(raw)
    return read_xp_continuous(raw)


class GaiaArchiveClient:
    """Minimal HTTPS session for ESA Gaia Archive TAP/ADQL and DataLink endpoints."""

    def __init__(
        self,
        *,
        archive_server: str = DEFAULT_ARCHIVE_SERVER,
        username: str | None = None,
        password: str | None = None,
        credentials_file: str | Path | None = None,
        timeout: float = 120.0,
    ) -> None:
        base = archive_server.rstrip("/") + "/"
        self.base_url = base
        self.timeout = float(timeout)
        self._jar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self._jar))
        self._logged_in = False

        if credentials_file is not None and (username is not None or password is not None):
            raise ValueError("Provide either username/password or credentials_file, not both")
        if credentials_file is not None:
            username, password = _read_credentials_file(credentials_file)
        if username is not None and password is not None:
            self.login(username, password)

    def _request(
        self,
        url: str,
        *,
        data: dict[str, str] | list[tuple[str, str]] | None = None,
    ) -> tuple[bytes, str]:
        encoded = None if data is None else urllib.parse.urlencode(data).encode("utf-8")
        req = urllib.request.Request(url, data=encoded, method="POST" if encoded is not None else "GET")
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                ctype = resp.headers.get("Content-Type", "")
                return resp.read(), ctype
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"Gaia Archive HTTP {exc.code} from {url}: {body}") from exc

    def login(self, username: str, password: str) -> None:
        """Authenticate against ESA Cosmos / Gaia TAP CAS."""
        login_url = urllib.parse.urljoin(self.base_url, "tap-server/login")
        self._request(login_url, data={"username": username, "password": password})
        self._logged_in = True

    def fetch_source_ids(
        self,
        source_ids,
        *,
        release: str | None = "dr3",
        retrieval_type: str = "XP_CONTINUOUS",
        data_structure: str = "COMBINED",
        format: str = "csv",
        data_release: str | None = None,
        chunk_size: int = 5000,
    ) -> XPBatch:
        """Download continuous XP spectra from Gaia DataLink for integer source IDs."""
        from .io import concat_batches

        ids = validate_source_ids(source_ids)
        rel_str = data_release or _archive_release_label(release)
        prefix = f"{rel_str} "
        dl_url = urllib.parse.urljoin(self.base_url, "data-server/datalink/links")

        batches: list[XPBatch] = []
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start : start + chunk_size]
            params: list[tuple[str, str]] = [
                ("RETRIEVAL_TYPE", retrieval_type),
                ("DATA_STRUCTURE", data_structure),
                ("FORMAT", format),
            ]
            for sid in chunk:
                params.append(("ID", f"{prefix}{sid}"))
            raw, ctype = self._request(dl_url, data=params)
            batch = _decode_archive_payload(raw, ctype)
            if len(batch) > 0:
                batches.append(batch)
        if not batches:
            raise ValueError(f"No {retrieval_type} data found for the given source IDs.")
        return concat_batches(batches)

    def fetch_adql(
        self,
        query: str,
        *,
        release: str | None = "dr3",
        retrieval_type: str = "XP_CONTINUOUS",
        data_structure: str = "COMBINED",
        data_release: str | None = None,
        format: str = "csv",
    ) -> XPBatch:
        """Execute a synchronous TAP ADQL query and return ``XPBatch``."""
        if not isinstance(query, str) or not query.strip().lower().startswith("select"):
            raise ValueError("ADQL query must start with SELECT")
        tap_url = urllib.parse.urljoin(self.base_url, "tap-server/tap/sync")
        raw, ctype = self._request(
            tap_url,
            data={
                "REQUEST": "doQuery",
                "LANG": "ADQL",
                "FORMAT": format,
                "QUERY": query.strip(),
            },
        )
        # Inspect columns: either full XP columns already selected, or source_id list for DataLink.
        try:
            return _decode_archive_payload(raw, ctype)
        except KeyError:
            # Query returned source_ids (e.g. from gaia_source), fetch XP via DataLink.
            import io

            import pyarrow.csv as pacsv

            raw_dec = raw
            if raw_dec[:2] == b"\x1f\x8b":
                import gzip

                raw_dec = gzip.decompress(raw_dec)
            tbl = pacsv.read_csv(io.BytesIO(raw_dec))
            lower = {c.lower(): c for c in tbl.column_names}
            if "source_id" not in lower:
                raise ValueError("ADQL query result must contain 'source_id' or XP coefficient columns")
            sids = tbl[lower["source_id"]].to_pylist()
            return self.fetch_source_ids(
                sids,
                release=release,
                retrieval_type=retrieval_type,
                data_structure=data_structure,
                format=format,
                data_release=data_release,
            )


def fetch_source_ids(
    source_ids,
    *,
    release: str | None = "dr3",
    username: str | None = None,
    password: str | None = None,
    credentials_file: str | Path | None = None,
    archive_server: str = DEFAULT_ARCHIVE_SERVER,
    retrieval_type: str = "XP_CONTINUOUS",
    data_release: str | None = None,
    format: str = "csv",
) -> XPBatch:
    """Fetch Gaia Archive XP spectra by source IDs without ``gaiaxpy`` or ``astroquery``."""
    client = GaiaArchiveClient(
        archive_server=archive_server,
        username=username,
        password=password,
        credentials_file=credentials_file,
    )
    return client.fetch_source_ids(
        source_ids,
        release=release,
        retrieval_type=retrieval_type,
        format=format,
        data_release=data_release,
    )


def fetch_adql(
    query: str,
    *,
    release: str | None = "dr3",
    username: str | None = None,
    password: str | None = None,
    credentials_file: str | Path | None = None,
    archive_server: str = DEFAULT_ARCHIVE_SERVER,
    retrieval_type: str = "XP_CONTINUOUS",
    data_release: str | None = None,
    format: str = "csv",
) -> XPBatch:
    """Run an ADQL query against ESA Gaia TAP/DataLink without ``gaiaxpy`` or ``astroquery``."""
    client = GaiaArchiveClient(
        archive_server=archive_server,
        username=username,
        password=password,
        credentials_file=credentials_file,
    )
    return client.fetch_adql(
        query,
        release=release,
        retrieval_type=retrieval_type,
        data_release=data_release,
        format=format,
    )
