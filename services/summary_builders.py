# --- Add/Replace these imports at the top of the file ---
import requests
import pandas as pd
import re
from typing import Optional, Dict, Any
import json, time, random
from pathlib import Path

BASE = "https://api.jolpi.ca/ergast/f1"
RAW = Path("data/raw/jolpi")
RAW.mkdir(parents=True, exist_ok=True)
sess = requests.Session()
# -----------------------------
ERGAST_BASE = "https://api.jolpi.ca/ergast/f1"  # mirror of ergast.com/api/f1
OPENF1 = "https://api.openf1.org/v1"
# Consider a status as "finished" if it matches these patterns
FINISH_PATTERNS = (
    r"^Finished$",
    r"^\+?\d+\s+Laps?$",     # "+1 Lap", "2 Laps"
    r"^\d+\s+Laps?$",        # "58 Laps" (Ergast style)
    r"^\+?\d+.*s$",          # "+10.123s"
    r"^Time$", 
    r"^Lapped$"
)
ALIASES_COUNTRY = {
    "singapore":"singapore",
    "italy":"italy","italie":"italy",
    "emilia":"italy","imola":"italy",
    "netherlands":"netherlands","dutch":"netherlands","zandvoort":"netherlands",
    "bahrain":"bahrain","sakhir":"bahrain",
    "spain":"spain","espagne":"spain","barcelona":"spain","catalunya":"spain",
    "usa":"united states","austin":"united states","americas":"united states",
    "miami":"miami","vegas":"las vegas","las":"las vegas",
    "japan":"japan","suzuka":"japan",
    "saudi":"saudi arabia","jeddah":"saudi arabia",
    "uae":"abu dhabi","yas":"abu dhabi",
    "monaco":"monaco",
    "australia":"australia","melbourne":"australia",
    "canada":"canada","montreal":"canada",
    "mexico":"mexico","cdmx":"mexico",
    "qatar":"qatar","losail":"qatar",
    "austria":"austria","spielberg":"austria","redbull ring":"austria",
    "hungary":"hungary","budapest":"hungary",
    "belgium":"belgium","spa":"belgium",
    "uk":"great britain","britain":"great britain","silverstone":"great britain",
    "azerbaijan":"azerbaijan","baku":"azerbaijan",
    "brasilia": "brazil", "brazil": "brazil", "brésil": "brazil"
}

ALIASES_MEETING = {
    "monza":"italian",
    "imola":"emilia romagna",
    "zandvoort":"dutch",
    "sakhir":"bahrain",
    "miami":"miami",
    "vegas":"las vegas","las vegas":"las vegas",
    "barcelona":"spanish","catalunya":"spanish",
    "jeddah":"saudi arabian",
    "suzuka":"japanese",
    "silverstone":"british",
    "spa":"belgian",
    "budapest":"hungarian",
    "spielberg":"austrian",
    "singapore":"singapore",
    "interlagos":"brazil"
}
_finish_re = re.compile("|".join(FINISH_PATTERNS), re.IGNORECASE)

def is_elimination(status: str) -> bool:
    """Return True if the status means DNF/DSQ/etc."""
    if not status:
        return True
    return not bool(_finish_re.search(str(status)))

def _ergast_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Simple GET with large limit to avoid pagination headaches."""
    url = f"{ERGAST_BASE}{path}"
    p = {"limit": 10000}
    if params:
        p.update(params)
    r = requests.get(url, params=p, timeout=30)
    r.raise_for_status()
    return r.json()

def get_driver_standings(year: int) -> pd.DataFrame:
    """
    Return official driver standings with points/wins.
    Columns: position, driver, code, constructor, points, wins_official
    """
    print("Fetching driver standings for year:", year)
    js = _ergast_get(f"/{year}/driverStandings.json")
    lists = js.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    if not lists:
        return pd.DataFrame(columns=["position","driver","code","constructor","points","wins_official"])

    standings = lists[0].get("DriverStandings", [])
    rows = []
    for d in standings:
        drv = d.get("Driver", {}) or {}
        cons = d.get("Constructors", [{}])[0] if d.get("Constructors") else {}
        rows.append({
            "position": int(d.get("position")) if d.get("position") else None,
            "driver": f"{drv.get('givenName','')} {drv.get('familyName','')}".strip(),
            "code": drv.get("code"),
            "constructor": cons.get("name"),
            "points": float(d.get("points", 0.0)),
            "wins_official": int(d.get("wins", 0)),
        })
    return pd.DataFrame(rows).sort_values("position")

def get_constructor_standings(year: int) -> pd.DataFrame:
    """
    Return official constructor standings.
    Columns: position, constructor, points, wins, nationality
    """
    js = _ergast_get(f"/{year}/constructorStandings.json")
    lists = js.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    if not lists:
        return pd.DataFrame(columns=["position","constructor","points","wins","nationality"])

    standings = lists[0].get("ConstructorStandings", [])
    rows = []
    for c in standings:
        cons = c.get("Constructor", {}) or {}
        rows.append({
            "position": int(c.get("position")) if c.get("position") else None,
            "constructor": cons.get("name"),
            "points": float(c.get("points", 0.0)),
            "wins": int(c.get("wins", 0)),
            "nationality": cons.get("nationality"),
        })
    return pd.DataFrame(rows).sort_values("position")



def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def cache_path(name: str) -> Path:
    safe = name.strip('/').replace('/', '_').replace('?', '_').replace('&', '_').replace('=', '_')
    return RAW / f"{safe}.json"

def get_paged(endpoint: str, params: Optional[Dict[str, Any]] = None,
              limit: int = 500, sleep: float = 0.5,
              max_retries: int = 6, backoff_base: float = 1.8,
              use_cache: bool = True) -> Dict[str, Any]:
    """
    Jolpi/Ergast pagination with rate-limit handling (429) and retries.
    Returns merged MRData with concatenated table rows and caches to disk.
    """
    params = dict(params or {})
    params["limit"] = limit

    cpath = cache_path(f"{endpoint}?limit={limit}")
    if use_cache and cpath.exists():
        try:
            return load_json(cpath)
        except Exception:
            pass

    def _request_with_retry(offset: int) -> Dict[str, Any]:
        for attempt in range(max_retries):
            try:
                r = sess.get(f"{BASE}{endpoint}", params={**params, "offset": offset}, timeout=30)
                if r.status_code == 429:
                    ra = r.headers.get("Retry-After")
                    wait_s = float(ra) if ra and ra.isdigit() else backoff_base ** attempt
                    wait_s += random.uniform(0, 0.5)
                    time.sleep(wait_s)
                    continue
                if 500 <= r.status_code < 600:
                    wait_s = backoff_base ** attempt + random.uniform(0, 0.5)
                    time.sleep(wait_s)
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException:
                wait_s = backoff_base ** attempt + random.uniform(0, 0.5)
                time.sleep(wait_s)
                continue
        r = sess.get(f"{BASE}{endpoint}", params={**params, "offset": offset}, timeout=30)
        r.raise_for_status()
        return r.json()

    data = _request_with_retry(0)
    mr = data.get("MRData", {})
    table_key = next((k for k in ("RaceTable","StandingsTable","CircuitTable","DriverTable",
                                  "ConstructorTable","StatusTable","LapTable") if k in mr), None)
    if not table_key:
        save_json(cpath, data)
        return data
    tbl = mr[table_key]
    rows_key = next((k for k in ("Races","StandingsLists","Circuits","Drivers",
                                 "Constructors","Status","Laps") if k in tbl), None)
    if not rows_key:
        save_json(cpath, data)
        return data

    merged = {"MRData": {**mr, table_key: {**tbl, rows_key: []}}}
    merged["MRData"][table_key][rows_key].extend(tbl.get(rows_key, []))

    total = int(mr.get("total", 0))
    lim   = int(mr.get("limit", limit))
    offset = lim

    while offset < total:
        time.sleep(sleep)
        page = _request_with_retry(offset)
        rows = page.get("MRData", {}).get(table_key, {}).get(rows_key, [])
        merged["MRData"][table_key][rows_key].extend(rows)
        offset += lim

    merged["MRData"]["total"] = str(len(merged["MRData"][table_key][rows_key]))
    merged["MRData"]["offset"] = "0"
    merged["MRData"]["limit"] = str(lim)

    save_json(cpath, merged)
    return merged

def get_season_results(year: int) -> pd.DataFrame:
    """
    Tous les résultats de course de l'année (toutes manches) pour calculs podiums/DNF/etc.
    """
    data = get_paged(f"/{year}/results.json", params={"limit": 1000})
    races = data["MRData"]["RaceTable"]["Races"]
    
    rows = []
    for r in races:
        rnd = int(r["round"])
        race_name = r["raceName"]
        date = r.get("date")
        for res in r.get("Results", []):
            drv = res["Driver"]
            cons = res["Constructor"]
            rows.append({
                "round": rnd,
                "race": race_name,
                "date": date,
                "driverId": drv["driverId"],
                "driver": f"{drv.get('givenName','')} {drv.get('familyName','')}".strip(),
                "code": drv.get("code"),
                "driver_number": drv.get("permanentNumber"),
                "constructor": cons["name"],
                "position": int(res["position"]) if "position" in res else None,
                "positionText": res.get("positionText"),
                "points": float(res.get("points", 0.0)),
                "status": res.get("status"),
                "grid": int(res.get("grid", 20)) if "grid" in res else None,
                "laps": int(res.get("laps", 0)) if "laps" in res else None,
                "positions_gained" : int(res["grid"]) -int(res["position"])  if "position" in res and "grid" in res else None,
                #"potential_points_win_grid" : 25 if int(res.get("grid", 0)) == 1 else (18 if int(res.get("grid", 0)) == 2 else (15 if int(res.get("grid", 0)) == 3 else (12 if int(res.get("grid", 0)) == 4 else (10 if int(res.get("grid", 0)) == 5 else (8 if int(res.get("grid", 0)) == 6 else (6 if int(res.get("grid", 0)) == 7 else (4 if int(res.get("grid", 0)) == 8 else (2 if int(res.get("grid", 0)) == 9 else (1 if int(res.get("grid", 0)) == 10 else 0))))))))) if "grid" in res else 0,
                "potential_points_gained": 25 - int(res["position"]) if int(res.get("grid",0)) == 1 else (18 - int(res["position"]) if int(res.get("grid",0)) == 2 else (15 - int(res["position"]) if int(res.get("grid",0)) == 3 else (12 - int(res["position"]) if int(res.get("grid",0)) == 4 else (10 - int(res["position"]) if int(res.get("grid",0)) == 5 else (8 - int(res["position"]) if int(res.get("grid",0)) == 6 else (6 - int(res["position"]) if int(res.get("grid",0)) == 7 else (4 - int(res["position"]) if int(res.get("grid",0)) == 8 else (2 - int(res["position"]) if int(res.get("grid",0)) == 9 else (1 - int(res["position"]) if int(res.get("grid",0)) == 10 else 0))))))))) if "position" in res and "grid" in res else 0
            })
    return pd.DataFrame(rows)

# -----------------------------
# Season builders (real data)
# -----------------------------
def build_season_summary(year: int) -> dict:
    """
    Returns:
      meta: {year, races_count}
      drivers: DataFrame with extended stats
      constructors: DataFrame with standings + cumulated podiums/elims
    """
    standings = get_driver_standings(year)
    results = get_season_results(year)
    constructors = get_constructor_standings(year)

    # Early exit if nothing
    if standings.empty and results.empty and constructors.empty:
        return {"meta": {"year": year, "races_count": 0}, "drivers": pd.DataFrame(), "constructors": pd.DataFrame()}

    # Compute derived stats from results
    df = standings.copy()
    if not results.empty:
        res = results.copy()
        # helpers:
        res["win"] = res["positionText"].astype(str).eq("1")
        res["podium"] = res["positionText"].astype(str).isin(["1","2","3"])
        res["elimination"] = res["status"].apply(is_elimination)
        res["grid"] = res["grid"].replace(0, 20)  # treat 0 as back of grid

        agg = (
            res.groupby(["driverId","driver","code","constructor"], as_index=False)
               .agg(
                   wins_calc=("win","sum"),
                   podiums=("podium","sum"),
                   eliminations=("elimination","sum"),
                   races=("round","nunique"),
                   points_sum=("position","count"),  # placeholder (Ergast doesn’t carry season points per result)
                   best_finish=("position","min"),
                   mean_pos=("position","mean"),
                   sd_pos=("position","std"),
                   mean_quali=("grid","mean"),
                   sd_quali=("grid","std"),
               )
        )

        # merge with official standings
        df = df.merge(agg, on=["driver","code","constructor"], how="left")
        # points: keep official
        df["points"] = df["points"].astype(float)
        # wins: prefer official if present
        df["wins"] = df["wins_official"].fillna(df["wins_calc"]).astype("Int64")
        # tidy types
        for c in ["podiums","eliminations","best_finish"]:
            if c in df:
                df[c] = df[c].astype("Int64")
        for c in ["mean_pos","sd_pos","mean_quali","sd_quali"]:
            if c in df:
                df[c] = df[c].astype("Float64")
        # team contribution
        if "constructor" in df and "points" in df and not df.empty:
            df["contribution_points"] = df["points"] / df.groupby("constructor")["points"].transform("sum")

    else:
        df.rename(columns={"wins_official":"wins"}, inplace=True)

    # Constructors: add cumulated podiums & eliminations from drivers
    cons = constructors.copy()
    if not df.empty and not cons.empty:
        temp_podiums = df.groupby("constructor")["podiums"].sum(min_count=1)
        temp_elim = df.groupby("constructor")["eliminations"].sum(min_count=1)
        cons = cons.merge(temp_podiums, on="constructor", how="left")
        cons = cons.merge(temp_elim, on="constructor", how="left", suffixes=("","_elim"))
        cons.rename(columns={"podiums":"podiums_cum","eliminations_elin":"eliminations_cum"}, inplace=True)
        if "eliminations" in cons.columns:
            cons.rename(columns={"eliminations":"eliminations_cum"}, inplace=True)
        cons["podiums_cum"] = cons["podiums_cum"].fillna(0).astype("Int64")
        cons["eliminations_cum"] = cons["eliminations_cum"].fillna(0).astype("Int64")
    races_count = results["round"].nunique() if not results.empty else 0
    meta = {"year": int(year), "races_count": int(races_count)}
    return {"meta": meta, "drivers": df.sort_values(["position","driver"]).reset_index(drop=True),
            "constructors": cons.sort_values("position").reset_index(drop=True)}

def build_season_summary_text(year: Optional[int]) -> dict:
    """
    Build text strings for drivers & constructors using real data.
    """
    if not year:
        return {"meta": {"year": None, "races_count": 0}, "drivers": "No driver data available.", "constructors": "No constructor data available."}

    season = build_season_summary(year)
    drivers_df = season["drivers"]
    constructors_df = season["constructors"]

    # Drivers text
    drivers_text = "=== Full Driver Standings ===\n"
    if isinstance(drivers_df, pd.DataFrame) and not drivers_df.empty:
        for _, r in drivers_df.sort_values("position").iterrows():
            drivers_text += (
                f"P{int(r['position'])} {r['driver']} ({r['constructor']}) — "
                f"{float(r['points']):.1f} pts, "
                f"Wins:{int(r.get('wins', 0))}, "
                f"Podiums:{int(r.get('podiums', 0)) if pd.notna(r.get('podiums')) else 0}, "
                f"Eliminations:{int(r.get('eliminations', 0)) if pd.notna(r.get('eliminations')) else 0}\n"
            )
    else:
        drivers_text += "No driver data available.\n"

    # Constructors text
    constructors_text = "=== Full Constructor Standings ===\n"
    if isinstance(constructors_df, pd.DataFrame) and not constructors_df.empty:
        for _, r in constructors_df.sort_values("position").iterrows():
            constructors_text += (
                f"P{int(r['position'])} {r['constructor']} — "
                f"{float(r['points']):.1f} pts, Wins:{int(r['wins'])}, "
                f"Podiums:{int(r.get('podiums_cum', 0))}, "
                f"Eliminations:{int(r.get('eliminations_cum', 0))}\n"
            )
    else:
        constructors_text += "No constructor data available.\n"

    return {
        "meta": season["meta"],
        "drivers": drivers_text.strip(),
        "constructors": constructors_text.strip()
    }

def summarize_season_fact(summary: dict) -> str:
    meta = summary.get("meta", {})
    year = meta.get("year", "N/A")
    races_count = meta.get("races_count", 0) or 0
    return (
        f"=== Season Summary for {year} ===\n"
        f"Total Races: {races_count}\n\n"
        f"{summary.get('drivers','')}\n\n"
        f"{summary.get('constructors','')}\n"
        f"==============================\n"
    )

def summarize_plain(summary: dict) -> str:
    m = summary["meta"]
    lines = []
    lines.append(f"{m.get('meeting_name','')} {m.get('year','')}")
    lines.append(f"{m.get('country_name','')} • {m.get('location','')} • {m.get('circuit_short_name','')} • {m.get('session_type','')}")
    if summary["top3"]:
        podium = " / ".join([f"P{r['position']} {r['full_name']} ({r['team_name']})" for r in summary["top3"]])
        lines.append(f"Podium: {podium}")
    if summary["grid_p1"]:
        lines.append(f"Grid P1: {summary['grid_p1']['name']} ({summary['grid_p1']['team']})")
    if summary["fastest_lap"]:
        fl = summary["fastest_lap"]
        lines.append(f"Fastest lap: {fl['name']} ({fl['team']}) (lap {fl['lap_number']}, {fl['lap_time_s']:.3f}s)")
    if summary["dnf_count"]:
        names = ", ".join([d.get("full_name","") for d in summary["dnf_list"]])
        lines.append(f"DNFs ({summary['dnf_count']}): {names}")
    if summary["classification"] is not None and not summary["classification"].empty:
        lines.append("Full classification of the drivers:")
        df = summary["classification"][["position","full_name"]]
        #transform df in one row 1 :  name, 2 : name, ...
        #if nan put 20 places
        df = df.fillna(20)
        classif = " | ".join([f"{int(r['position'])}: {r['full_name']}" for _, r in df.iterrows()])
        lines.append(classif)

    return "\n".join(lines)

def is_season_query(q: str) -> Optional[int]:
    ql = q.lower()
    m = re.search(r"\b(20\d{2})\b", ql)
    year = int(m.group(1)) if m else None
    if ("saison" in ql or "season" in ql or "" in ql or "" in ql) and year:
        return year
    return None


def is_gp_query(q: str) -> bool:
    ql = q.lower()
    for t in list(ALIASES_COUNTRY.keys()) + list(ALIASES_MEETING.keys()):
        if t in ql:
            return True
    return False
        
def is_year_query(q: str) -> Optional[int]:
    ql = q.lower()
    m = re.search(r"\b(20\d{2})\b", ql)
    #return true if year is found and false otherwise
    year = int(m.group(1)) if m else None
    return True if year else False
def find_session_from_query(query: str, prefer="race"):
    """
    Trouve la bonne session à partir d'une requête libre.
    - retire des stopwords (grand/prix/de/…)
    - fabrique des 'hints' pays/meeting/ville-circuit
    - score chaque session plutôt que de faire un simple contains()
    """
    q = query.strip().lower()
    year = _extract_year(q)
    tokens = _clean_tokens(q)
    want_sprint = "sprint" in q
    target_type = "Sprint" if want_sprint else "Race"
    # Hints
    hint_country, hint_meeting, hint_citycirc = _hints(tokens)
    print("The year extracted is", year)
    # Fetch sessions (filtre année si présent)
    params = {"year": year} if year else {}
    df = pd.DataFrame(_get("sessions", **params))
    print("Here are the tokens", tokens)
    print("Here the df", df)
    if df.empty:
        raise ValueError("No sessions from OpenF1 for given filters.")
    # Filtrer par type
    if "session_type" in df.columns:
        df = df[df["session_type"].str.lower() == target_type.lower()]
    else:
        df = df[df["session_name"].str.contains(target_type, case=False, na=False)]
    if df.empty:
        raise ValueError(f"No {target_type} sessions found.")

    # Colonnes utiles
    for c in ["meeting_name","country_name","location","circuit_short_name","year","date_start"]:
        if c not in df.columns:
            df[c] = ""

    # Pré-score via hints forts (country/meeting) si présents
    def score_row(row):
        s = 0
        meeting = str(row["meeting_name"]).lower()
        country = str(row["country_name"]).lower()
        location = str(row["location"]).lower()
        circuit  = str(row["circuit_short_name"]).lower()

        # Hints pays -> gros bonus
        for hc in hint_country:
            if hc and hc in country: s += 30

        # Hints meeting (Italian, Bahrain, etc.)
        for hm in hint_meeting:
            if hm and hm in meeting: s += 20

        # Ville/circuit tokens
        for t in hint_citycirc:
            if t in location: s += 12
            if t in circuit:  s += 10
            if t in meeting:  s += 8
            if t in country:  s += 6

        # Tokens restants (faible poids) — évite que "grand/prix" pollue
        for t in tokens:
            if t in {"sprint"}:  # déjà pris en compte
                continue
            if t in meeting:  s += 2
            if t in country:  s += 2
            if t in location: s += 2
            if t in circuit:  s += 2

        # Bonus si année exacte
        try:
            if year and int(row.get("year", 0)) == year:
                s += 5
        except Exception:
            pass

        return s

    df = df.copy()
    df["__score__"] = df.apply(score_row, axis=1)

    # Si on a un hint pays fort, on peut éliminer le reste (hard filter)
    if hint_country:
        mask_country = False
        for hc in hint_country:
            mask_country = mask_country | df["country_name"].str.lower().str.contains(hc, na=False)
        if mask_country.any():
            df = df[mask_country]

    # Tri: score desc, puis année desc, puis date_start asc
    if "year" in df.columns and "date_start" in df.columns:
        df = df.sort_values(["__score__", "year", "date_start"], ascending=[False, False, True])
    else:
        df = df.sort_values("__score__", ascending=False)

    if df.empty or df.iloc[0]["__score__"] <= 0:
        # dernier recours: filter strict sur 'singapore' etc.
        for t in list(hint_country) + list(hint_meeting) + list(hint_citycirc):
            strict = df[
                df["meeting_name"].str.lower().str.contains(t, na=False) |
                df["country_name"].str.lower().str.contains(t, na=False) |
                df["location"].str.lower().str.contains(t, na=False) |
                df["circuit_short_name"].str.lower().str.contains(t, na=False)
            ]
            if not strict.empty:
                df = strict
                break

    if df.empty:
        raise ValueError(f"No matching {target_type} found for query: '{query}'")

    row = df.iloc[0].to_dict()
    print(f"Selected session: {row['session_key']} ({row['meeting_name']} {row['year']}) with score {row['__score__']}")

    return {
        "session_key": int(row["session_key"]),
        "meeting_key": int(row["meeting_key"]),
        "meeting_name": row.get("meeting_name") or "",
        "country_name": row.get("country_name") or "",
        "location": row.get("location") or "",
        "circuit_short_name": row.get("circuit_short_name") or "",
        "year": int(row.get("year")) if pd.notna(row.get("year")) else None,
        "session_type": target_type,
        "debug_top_hits": df.head(5)[
            ["__score__","year","meeting_name","country_name","location","circuit_short_name","session_key"]
        ].to_dict(orient="records"),  # utile pour debugger
    }
def build_gp_summary(query: str) -> dict:
    meta = find_session_from_query(query)
    k = meta["session_key"]

    table = get_classification(k)
    top3 = table.head(3)[["position","full_name","team_name"]].to_dict(orient="records") if not table.empty else []

    grid_p1 = get_grid_p1(k)
    fl = get_fastest_lap(k)
    dnfs = get_dnfs(k)
    print(table)
    return {
        "meta": meta,
        "top3": top3,
        "grid_p1": grid_p1,
        "fastest_lap": fl,
        "dnf_count": int(dnfs.shape[0]) if not dnfs.empty else 0,
        "dnf_list": dnfs.to_dict(orient="records") if not dnfs.empty else [],
        "classification": table,  # full dataframe
    }

def _extract_year(text: str, default=None):
    m = re.search(r"\b(20\d{2})\b", text)
    return int(m.group(1)) if m else default

STOPWORDS = {
    "grand","prix","gp","g","de","du","des","le","la","les","the",
    "race","résumé","resume","summary","round","of","formula","f1"
}

def _clean_tokens(q: str):
    toks = [t for t in re.split(r"[^a-z0-9]+", q.lower()) if t]
    return [t for t in toks if t not in STOPWORDS and not t.isdigit()]

def _hints(tokens):
    country = set()
    meeting = set()
    city_or_circuit = set()
    for t in tokens:
        if t in ALIASES_COUNTRY: country.add(ALIASES_COUNTRY[t])
        if t in ALIASES_MEETING: meeting.add(ALIASES_MEETING[t])
        # brut: tout token non mappé est potentiellement une ville/circuit
        if t not in ALIASES_COUNTRY and t not in ALIASES_MEETING:
            city_or_circuit.add(t)
    return country, meeting, city_or_circuit



def _fetch(endpoint: str, **params):
    r = requests.get(f"{OPENF1}/{endpoint}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def _clean_params(params: dict) -> dict:
    # Supprime None / "" / [] / {} et force str sur dates
    out = {}
    for k, v in (params or {}).items():
        if v in (None, "", [], {}):
            continue
        if hasattr(v, "strftime"):  # datetime.date/datetime
            v = v.strftime("%Y-%m-%d")
        out[str(k)] = str(v)
    return out

def _get(endpoint: str, **params):
    r = requests.get(f"{OPENF1}/{endpoint}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def get_classification(session_key: int) -> pd.DataFrame:
    res = pd.DataFrame(_get("session_result", session_key=session_key))
    drv = pd.DataFrame(_get("drivers", session_key=session_key))
    if not res.empty and not drv.empty:
        res = res.merge(
            drv[["driver_number", "full_name", "name_acronym", "team_name"]],
            on="driver_number", how="left"
        )
        res = res.sort_values("position")
    return res

def get_grid_p1(session_key: int):
    """Pole (grid P1 réel) via /starting_grid."""
    meeting = pd.DataFrame(_fetch("sessions", session_key=session_key)).iloc[0]["meeting_key"]
    grid = pd.DataFrame(_fetch("starting_grid", meeting_key=meeting))
    if grid.empty:
        return None
    # On récupère le nom
    drv = pd.DataFrame(_fetch("drivers", session_key=session_key))
    g = grid.iloc[0].to_dict()
    name = None
    team = None
    if not drv.empty:
        row = drv[drv["driver_number"] == g["driver_number"]].iloc[0]
        name = row.get("full_name")
        team = row.get("team_name")
    return {"driver_number": g["driver_number"], "name": name, "team": team}


def get_fastest_lap(session_key: int):
    laps = pd.DataFrame(_get("laps", session_key=session_key))
    if laps.empty or "lap_duration" not in laps.columns:
        return None
    laps = laps.dropna(subset=["lap_duration"])
    laps["lap_duration"] = laps["lap_duration"].astype(float)
    idx = laps["lap_duration"].idxmin()
    best = laps.loc[idx]
    drv = pd.DataFrame(_get("drivers", session_key=session_key))
    name = team = None
    if not drv.empty:
        r = drv[drv["driver_number"] == int(best["driver_number"])]
        if not r.empty:
            rr = r.iloc[0]
            name, team = rr.get("full_name"), rr.get("team_name")
    return {
        "driver_number": int(best["driver_number"]),
        "name": name,
        "team": team,
        "lap_number": int(best["lap_number"]),
        "lap_time_s": float(best["lap_duration"]),
    }

def get_dnfs(session_key: int) -> pd.DataFrame:
    """Liste des abandons via le flag dnf de /session_result."""
    res = pd.DataFrame(_fetch("session_result", session_key=session_key))
    if res.empty:
        return res
    dnfs = res[res["dnf"] == True].copy()
    if dnfs.empty:
        return dnfs
    drv = pd.DataFrame(_fetch("drivers", session_key=session_key))
    if not drv.empty:
        dnfs = dnfs.merge(
            drv[["driver_number", "full_name", "team_name"]],
            on="driver_number", how="left"
        )
        #if dns then retirement_reason = dns, if dnf then retirement_reason = dnf, if dnq then retirement_reason = dnq
        dnfs["retirement_reason"] = dnfs.apply(
            lambda row: "DNS" if row.get("dns") else ("DNF" if row.get("dnf") else ("DNQ" if row.get("dnq") else "Unknown")),
            axis=1
        )
    return dnfs[["position", "driver_number", "full_name", "team_name", "number_of_laps", "retirement_reason"]].sort_values(
        by=["position", "number_of_laps"], ascending=[True, False]
    )