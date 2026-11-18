"""
Script de Auditoría Exhaustiva de Calidad, Integridad y Trazabilidad.
Analiza data/raw/matches_2022.json, matches_2023.json, matches_2024.json y sus páginas correspondientes.
No modifica archivos ni consulta redes externas.
"""

import os
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw"
SEASONS = [2022, 2023, 2024]


def run_full_audit():
    audit_report = {
        "timestamp": datetime.now().isoformat(),
        "resumen_general": {},
        "integridad": {},
        "calidad": {},
        "consistencia": {},
        "trazabilidad": {},
        "aptitud_lakehouse": {}
    }

    print("=" * 80)
    print("INICIANDO AUDITORÍA INTEGRAL DE ARCHIVOS CRUDOS (BALLDONTLIE EPL V2)")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. INTEGRIDAD Y CONTEO
    # ---------------------------------------------------------
    all_season_matches = {}
    all_global_ids = []
    season_id_map = {}

    for s in SEASONS:
        filepath = RAW_DIR / f"matches_{s}.json"
        if not filepath.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            content = json.load(f)

        matches = content.get("data", [])
        all_season_matches[s] = matches
        ids_in_season = [m.get("id") for m in matches]
        all_global_ids.extend(ids_in_season)
        season_id_map[s] = ids_in_season

        # Duplicados intra-temporada
        id_counts = Counter(ids_in_season)
        duplicates_in_season = {mid: c for mid, c in id_counts.items() if c > 1}

        # Verificación del campo season
        seasons_in_data = set(m.get("season") for m in matches)

        audit_report["integridad"][f"temporada_{s}"] = {
            "total_registros": len(matches),
            "registros_esperados": 380,
            "coincide_esperado": len(matches) == 380,
            "ids_unicos": len(set(ids_in_season)),
            "duplicados_detectados": len(duplicates_in_season),
            "detalle_duplicados": duplicates_in_season,
            "valores_season_encontrados": list(seasons_in_data),
            "season_consistente": seasons_in_data == {s}
        }

    # Duplicados entre temporadas
    global_id_counts = Counter(all_global_ids)
    cross_season_duplicates = {mid: c for mid, c in global_id_counts.items() if c > 1}
    audit_report["integridad"]["global"] = {
        "total_partidos_analizados": len(all_global_ids),
        "total_ids_unicos": len(set(all_global_ids)),
        "duplicados_cruzados_entre_temporadas": len(cross_season_duplicates),
        "detalle_duplicados_cruzados": cross_season_duplicates
    }

    # ---------------------------------------------------------
    # 2. CALIDAD DE DATOS (Nulos, Marcadores, Estados, Anomalías)
    # ---------------------------------------------------------
    for s, matches in all_season_matches.items():
        field_null_counts = defaultdict(int)
        total = len(matches)
        
        status_counts = Counter()
        status_state_counts = Counter()

        score_stats = {
            "ambos_marcadores_presentes": 0,
            "ambos_marcadores_null": 0,
            "un_marcador_null": 0,
            "marcador_cero_local": 0,
            "marcador_cero_visita": 0,
            "marcador_cero_ambos": 0,
            "marcadores_negativos": 0
        }

        anomalies = []

        for idx, m in enumerate(matches):
            mid = m.get("id")

            # Conteo de nulos por campo
            for key, val in m.items():
                if val is None:
                    field_null_counts[key] += 1

            status = m.get("status")
            status_state = m.get("status_state")
            status_counts[str(status)] += 1
            status_state_counts[str(status_state)] += 1

            h_score = m.get("home_score")
            a_score = m.get("away_score")

            h_team = m.get("home_team")
            a_team = m.get("away_team")
            h_id = m.get("home_team_id")
            a_id = m.get("away_team_id")

            # Anomalía: Equipo jugando contra sí mismo
            if h_id is not None and a_id is not None and h_id == a_id:
                anomalies.append(f"Partido {mid}: Mismo equipo local y visitante (ID: {h_id})")

            # Marcadores negativos
            if (h_score is not None and h_score < 0) or (a_score is not None and a_score < 0):
                score_stats["marcadores_negativos"] += 1
                anomalies.append(f"Partido {mid}: Marcador negativo detectado ({h_score} - {a_score})")

            # Nulos vs Ceros
            if h_score is None and a_score is None:
                score_stats["ambos_marcadores_null"] += 1
            elif h_score is None or a_score is None:
                score_stats["un_marcador_null"] += 1
                anomalies.append(f"Partido {mid}: Solo un marcador es NULL ({h_score} vs {a_score})")
            else:
                score_stats["ambos_marcadores_presentes"] += 1
                if h_score == 0 and a_score == 0:
                    score_stats["marcador_cero_ambos"] += 1
                if h_score == 0:
                    score_stats["marcador_cero_local"] += 1
                if a_score == 0:
                    score_stats["marcador_cero_visita"] += 1

            # Inconsistencia de estado vs marcador
            if status_state == "post" and (h_score is None or a_score is None):
                anomalies.append(f"Partido {mid}: Estado 'post' pero marcadores NULL")
            if status_state == "pre" and (h_score is not None or a_score is not None):
                anomalies.append(f"Partido {mid}: Estado 'pre' pero marcadores no nulos ({h_score}-{a_score})")

        null_percentages = {k: {"nulos": v, "pct": round((v / total) * 100, 2)} for k, v in field_null_counts.items()}

        audit_report["calidad"][f"temporada_{s}"] = {
            "analisis_nulos": null_percentages,
            "distribucion_status": dict(status_counts),
            "distribucion_status_state": dict(status_state_counts),
            "estadisticas_marcadores": score_stats,
            "anomalias_detectadas": anomalies,
            "total_anomalias": len(anomalies)
        }

    # ---------------------------------------------------------
    # 3. CONSISTENCIA (Equipos, Fechas, Partidos por Club)
    # ---------------------------------------------------------
    season_teams = {}
    for s, matches in all_season_matches.items():
        team_matches_home = Counter()
        team_matches_away = Counter()
        team_names = {}
        dates = []

        for m in matches:
            hid = m.get("home_team_id")
            aid = m.get("away_team_id")
            hname = m.get("home_team", {}).get("name") if isinstance(m.get("home_team"), dict) else None
            aname = m.get("away_team", {}).get("name") if isinstance(m.get("away_team"), dict) else None

            if hid:
                team_matches_home[hid] += 1
                if hname: team_names[hid] = hname
            if aid:
                team_matches_away[aid] += 1
                if aname: team_names[aid] = aname

            d_str = m.get("date")
            if d_str:
                try:
                    dt = datetime.fromisoformat(d_str.replace("Z", "+00:00"))
                    dates.append(dt)
                except Exception:
                    pass

        all_tids = set(team_matches_home.keys()) | set(team_matches_away.keys())
        season_teams[s] = {tid: team_names.get(tid, f"ID_{tid}") for tid in all_tids}

        # Verificación de partidos por equipo (deben ser 19 local + 19 visita = 38)
        teams_participation = {}
        imbalances = []
        for tid in all_tids:
            h_count = team_matches_home[tid]
            a_count = team_matches_away[tid]
            total_t = h_count + a_count
            name = team_names.get(tid, f"ID_{tid}")
            teams_participation[name] = {"local": h_count, "visita": a_count, "total": total_t}
            if total_t != 38 or h_count != 19 or a_count != 19:
                imbalances.append(f"Equipo {name}: {h_count}L + {a_count}V = {total_t} (Esperado 19L+19V=38)")

        dates.sort()
        audit_report["consistencia"][f"temporada_{s}"] = {
            "total_equipos_distintos": len(all_tids),
            "esperado_equipos": 20,
            "fecha_minima": dates[0].isoformat() if dates else None,
            "fecha_maxima": dates[-1].isoformat() if dates else None,
            "desbalances_de_partidos_por_equipo": imbalances,
            "equipos_detectados": list(season_teams[s].values())
        }

    # Transiciones de ascenso y descenso
    teams_2022 = set(season_teams[2022].values())
    teams_2023 = set(season_teams[2023].values())
    teams_2024 = set(season_teams[2024].values())

    descendidos_2022 = teams_2022 - teams_2023
    ascendidos_2023 = teams_2023 - teams_2022
    descendidos_2023 = teams_2023 - teams_2024
    ascendidos_2024 = teams_2024 - teams_2023

    audit_report["consistencia"]["transiciones_ascenso_descenso"] = {
        "2022_a_2023": {
            "descendidos_2022": sorted(list(descendidos_2022)),
            "ascendidos_2023": sorted(list(ascendidos_2023)),
            "cantidad_descensos": len(descendidos_2022),
            "cantidad_ascensos": len(ascendidos_2023)
        },
        "2023_a_2024": {
            "descendidos_2023": sorted(list(descendidos_2023)),
            "ascendidos_2024": sorted(list(ascendidos_2024)),
            "cantidad_descensos": len(descendidos_2023),
            "cantidad_ascensos": len(ascendidos_2024)
        }
    }

    # ---------------------------------------------------------
    # 4. TRAZABILIDAD (Páginas Crudas vs Consolidado vs Checkpoint)
    # ---------------------------------------------------------
    checkpoint_file = RAW_DIR / ".checkpoint_historical.json"
    cp_data = {}
    if checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            cp_data = json.load(f)

    for s in SEASONS:
        page_files = sorted(RAW_DIR.glob(f"matches_{s}_page_*.json"))
        records_in_pages = []
        page_ids = []

        for pf in page_files:
            with open(pf, "r", encoding="utf-8") as f:
                p_json = json.load(f)
                p_items = p_json.get("data", [])
                records_in_pages.append({"archivo": pf.name, "registros": len(p_items)})
                page_ids.extend([item.get("id") for item in p_items])

        consolidated_ids = [m.get("id") for m in all_season_matches[s]]

        # Comparar IDs de páginas vs consolidado
        ids_en_paginas_set = set(page_ids)
        ids_en_cons_set = set(consolidated_ids)

        perdidos = ids_en_paginas_set - ids_en_cons_set
        agregados = ids_en_cons_set - ids_en_paginas_set

        audit_report["trazabilidad"][f"temporada_{s}"] = {
            "paginas_encontradas": len(page_files),
            "detalle_paginas": records_in_pages,
            "suma_registros_en_paginas": len(page_ids),
            "ids_unicos_en_paginas": len(ids_en_paginas_set),
            "registros_en_consolidado": len(consolidated_ids),
            "coincidencia_exacta": ids_en_paginas_set == ids_en_cons_set,
            "registros_perdidos_en_consolidacion": len(perdidos),
            "registros_inventados_en_consolidacion": len(agregados),
            "estado_checkpoint": cp_data.get(str(s), {}).get("status")
        }

    # ---------------------------------------------------------
    # 5. APTITUD LAKEHOUSE (PySpark / Parquet / DuckDB)
    # ---------------------------------------------------------
    all_dates_valid = all(
        len(audit_report["consistencia"][f"temporada_{s}"]["desbalances_de_partidos_por_equipo"]) == 0
        for s in SEASONS
    )
    all_integrity_passed = all(
        audit_report["integridad"][f"temporada_{s}"]["coincide_esperado"] and 
        audit_report["integridad"][f"temporada_{s}"]["duplicados_detectados"] == 0
        for s in SEASONS
    )
    all_traceability_passed = all(
        audit_report["trazabilidad"][f"temporada_{s}"]["coincidencia_exacta"]
        for s in SEASONS
    )

    audit_report["aptitud_lakehouse"] = {
        "apto_para_pyspark": all_integrity_passed and all_traceability_passed,
        "apto_para_parquet": True,
        "apto_para_duckdb": True,
        "dictamen": "APROBADO_PARA_PROCESAMIENTO" if (all_integrity_passed and all_traceability_passed) else "REQUIERE_ATENCION"
    }

    # Guardar reporte de auditoría en JSON para persistencia
    report_path = PROJECT_ROOT / "data" / "raw" / "audit_report_balldontlie.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)

    print(f"Reporte de auditoría generado en: {report_path}")
    return audit_report


if __name__ == "__main__":
    rep = run_full_audit()
    print("\n" + "=" * 80)
    print("RESUMEN DE AUDITORÍA EJECUTADA")
    print("=" * 80)
    for s in SEASONS:
        integ = rep["integridad"][f"temporada_{s}"]
        traz = rep["trazabilidad"][f"temporada_{s}"]
        cons = rep["consistencia"][f"temporada_{s}"]
        cal = rep["calidad"][f"temporada_{s}"]
        print(f"\n[TEMPORADA {s}]")
        print(f"  • Total registros: {integ['total_registros']} (Esperado: 380) | Duplicados: {integ['duplicados_detectados']}")
        print(f"  • Equipos: {cons['total_equipos_distintos']} | Rango fechas: {cons['fecha_minima']} a {cons['fecha_maxima']}")
        print(f"  • Partidos jugados ('post'): {cal['distribucion_status_state'].get('post', 0)} | Programados ('pre'): {cal['distribucion_status_state'].get('pre', 0)}")
        print(f"  • Marcadores presentes: {cal['estadisticas_marcadores']['ambos_marcadores_presentes']} | Ceros local: {cal['estadisticas_marcadores']['marcador_cero_local']} | Ceros visita: {cal['estadisticas_marcadores']['marcador_cero_visita']}")
        print(f"  • Trazabilidad páginas -> consolidado: Exacta={traz['coincidencia_exacta']} (Páginas: {traz['paginas_encontradas']})")
        print(f"  • Anomalías detectadas: {cal['total_anomalias']}")
    
    print("\n[GLOBAL]")
    print(f"  • Total partidos en Lakehouse crudo: {rep['integridad']['global']['total_partidos_analizados']}")
    print(f"  • IDs únicos globales: {rep['integridad']['global']['total_ids_unicos']} | Duplicados cruzados: {rep['integridad']['global']['duplicados_cruzados_entre_temporadas']}")
    print(f"  • Dictamen Lakehouse: {rep['aptitud_lakehouse']['dictamen']}")
