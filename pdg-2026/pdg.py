#!/usr/bin/env python3
"""
Forest Analysis: estimation of forest characteristics and growth ("accrescimenti").
CLI entry point and template processing.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from typing import cast

from natsort import natsort_keygen

from pdg.harvest_rules import max_harvest
from pdg.computation import (
    COL_COMPRESA, COL_PARTICELLA, COL_GENERE,
    COL_D_CM, COL_H_M, COL_V_M3, COL_PRESSLER, COL_L10_MM,
    COL_GOVERNO, GOV_FUSTAIA, GOV_CEDUO, COL_AREA_SAGGIO,
    calculate_all_trees_volume, compute_heights,
    fit_curves_from_ipsometro, fit_curves_from_originali, fit_curves_from_tabelle,
)
from pdg.io import load_csv, load_trees
from pdg.formatters import (
    OPT_STILE, fmt_num, set_decimal_comma,
    SnippetFormatter, HTMLSnippetFormatter, LaTeXSnippetFormatter, CSVSnippetFormatter,
)
from pdg.ceduo import (
    load_coppice_parcels, load_adjacencies, last_harvests_from_calendario,
    schedule_coppice,
)
from pdg.simulation import ORDINE_VOL_HA, ORDINE_VOL_TOT, ORDINE_DATA, write_volume_log
from pdg.core import (
    OPT_PER_COMPRESA, OPT_PER_PARTICELLA, OPT_PER_GENERE,
    OPT_STIME_TOTALI, OPT_TOTALI, OPT_METRICA,
    OPT_INTERV_FIDUC, OPT_SOLO_MATURE,
    OPT_COL_COMPARTO, OPT_COL_ETA, OPT_COL_AREA_HA,
    OPT_COL_VOLUME, OPT_COL_VOLUME_HA,
    OPT_COL_VOLUME_MATURE, OPT_COL_VOLUME_MATURE_HA,
    OPT_COL_PP_MAX, OPT_COL_PRELIEVO, OPT_COL_PRELIEVO_HA, OPT_COL_INCR_CORR,
    OPT_X_MAX, OPT_Y_MAX,
    OPT_ANNO_INIZIO, OPT_ANNO_FINE, OPT_INTERVALLO, OPT_INTERVALLO_ANNO,
    OPT_MORTALITA, OPT_PRUDENZA, OPT_RIDUZIONE, OPT_VOLUME_OBIETTIVO, OPT_CALENDARIO, OPT_ORDINE, OPT_PARTICELLE_MIN,
    parse_gap_overrides,
    OPT_COL_PRIMA_DOPO, OPT_PARTICELLE, OPT_ADIACENZE, OPT_EQUAZIONI,
    read_past_harvests, parcel_data,
    get_color_map,
    render_hypsometric_graph, render_diameter_class_graph, render_diameter_class_table,
    render_prop, render_prop_coppice,
    render_volume_table, render_harvest_table, render_harvest_plan,
    render_pct_growth_table, render_pct_growth_graph, render_coppice_schedule,
    skip_graphs,
)


# =============================================================================
# TEMPLATE DIRECTIVE PARSING
# =============================================================================

@dataclass
class Directive:
    """A parsed @@keyword(params) template directive."""
    keyword: str
    params: dict
    full_text: str

# Output format types
class Fmt:
    HTML = 'html'
    TEX  = 'tex'
    PDF  = 'pdf'
    CSV  = 'csv'

# Template directive keywords
class Dir:
    DIAMETER_CLASS_GRAPH = 'grafico_classi_diametriche'
    DIAMETER_CLASS_TABLE = 'tabella_classi_diametriche'
    HARVEST_PLAN = 'piano_di_taglio'
    HARVEST_TABLE = 'prelievi'
    HYPSOMETRIC_GRAPH = 'grafico_classi_ipsometriche'
    PARCELS = 'particelle'
    PCT_GROWTH_GRAPH = 'grafico_incremento_percentuale'
    PCT_GROWTH_TABLE = 'tabella_incremento_percentuale'
    COPPICE_SCHEDULE = 'calendario_ceduo'
    PROP = 'prop'
    PROP_CEDUO = 'prop_ceduo'
    VOLUME_TABLE = 'volumi'


def check_allowed_params(directive: str, params: dict, options: dict):
    """Check that all keys in params are in options, raise ValueError if not."""
    bad_keys = []
    for key in params.keys():
        if key not in options and key not in ['alberi', 'compresa', 'particella', 'genere']:
            bad_keys.append(key)
    if bad_keys:
        raise ValueError(f"Parametri non validi '{bad_keys}' in @@{directive}")


def check_required_params(directive: str, params: dict, required_keys: list[str]):
    """Check that all keys in required_keys are present in params, raise ValueError if not."""
    missing_keys = []
    for key in required_keys:
        if key not in params:
            missing_keys.append(key)
    if missing_keys:
        raise ValueError(f"Parametri obbligatori mancanti '{missing_keys}' in @@{directive}")


def check_param_values(options: dict, key: str, valid_values: list[str], directive: str):
    """Check that options[key] is in valid_values, raise ValueError if not."""
    value = options.get(key)
    if value not in valid_values:
        raise ValueError(f"Valore non valido per '{key}' in @@{directive}: '{value}'. "
                         f"Valori validi: {', '.join(valid_values)}")


def _bool_opt(params: dict, key: str, enabled: bool = True) -> bool:
    """Parse a si/no boolean option from template directive params."""
    return params.get(key, 'si' if enabled else 'no').lower() == 'si'


def parse_template_directive(line: str) -> Directive | None:
    """
    Parse a template directive like @@grafico_classi_ipsometriche(compresa=Serra, genere=Abete).

    Filter keys (compresa, particella, genere) are always lists (even single values):
        @@grafico_classi_diametriche(compresa=Serra) -> {'compresa': ['Serra']}
        @@grafico_classi_diametriche(compresa=Serra, compresa=Fabrizia) -> {'compresa': ['Serra', 'Fabrizia']}

    Other keys remain scalar values.

    Returns:
        Directive or None if not a valid directive
    """
    # Match pattern: @@keyword(param=value, param=value, ...)
    pattern = r'@@(\w+)\((.*?)\)'
    match = re.search(pattern, line)

    if not match:
        return None

    keyword = match.group(1)
    params_str = match.group(2)
    full_text = match.group(0)

    # Keys that should always be lists (filter parameters + file parameters)
    list_keys = {'compresa', 'particella', 'genere', 'alberi', OPT_EQUAZIONI,
                  OPT_INTERVALLO_ANNO}

    params = {}
    if params_str.strip():
        # Split by comma and parse key=value pairs
        for param in params_str.split(','):
            param = param.strip()
            if '=' in param:
                key, value = param.split('=', 1)
                key = key.strip()
                value = value.strip()

                if key in list_keys:
                    # Always accumulate into lists
                    if key not in params:
                        params[key] = []
                    params[key].append(value)
                else:
                    # Scalar value (last one wins if repeated)
                    params[key] = value

    return Directive(keyword=keyword, params=params, full_text=full_text)


# =============================================================================
# TEMPLATE PROCESSING
# =============================================================================

DIRECTIVE_PATTERN = re.compile(r'@@(\w+)\((.*?)\)')

def process_template(template_text: str, data_dir: Path,
                     parcel_file: str,
                     output_dir: Path,
                     format_type: str,
                     template_dir: Path | None = None,
                     log_simulazione: bool = False) -> str:
    """
    Process template by substituting @@directives with generated content.

    Args:
        template_text: Input template
        data_dir: Base directory for data files (alberi, equazioni)
        parcel_file: Parcel metadata file
        output_dir: Where to save generated graphs
        format_type: 'html' or 'tex'
        template_dir: Directory containing template files (for @@particelle modello)

    Returns:
        Processed template text
    """
    # Track filenames to make duplicates unique
    filename_counts = defaultdict(int)
    harvest_plan_count = 0
    def _build_graph_filename(comprese: list[str], particelle: list[str],
                              generi: list[str], keyword: str) -> str:
        """Build a filename for a graph based on the parameters (all lists)."""
        parts = [keyword]
        if comprese:
            parts.append('-'.join(sorted(comprese)))
        else:
            parts.append('tutte')
        parts.append('-'.join(sorted(particelle)))
        parts.append('-'.join(sorted(generi)))
        base_name = '_'.join(parts)
        filename_counts[base_name] += 1
        return f'{base_name}_{filename_counts[base_name]:02d}.png'

    def render_particelle(comprese: list[str], particelle: list[str],
                          particelle_df, params: dict):
        """
        Render information about all parcels in compresa by filling in a model template.
        """
        if len(comprese) != 1:
            raise ValueError("@@particelle richiede esattamente compresa=X")
        modello = params.get('modello')
        if not modello:
            raise ValueError("@@particelle richiede modello=BASENAME")
        if not template_dir:
            raise ValueError("@@particelle richiede --input per trovare il modello")

        match format_type:
            case Fmt.HTML: ext = '.html'
            case Fmt.TEX | Fmt.PDF: ext = '.tex'
            case Fmt.CSV: ext = '.csv'
            case _: raise ValueError(f"Formato non supportato per @@particelle: {format_type}")
        modello_path = template_dir / (modello + ext)
        if not modello_path.exists():
            raise ValueError(f"Modello non trovato: {modello_path}")
        with open(modello_path, 'r', encoding='utf-8') as f:
            modello_text = f.read()

        compresa = comprese[0]
        ceduo = _bool_opt(params, 'ceduo', enabled=False)
        governo = GOV_CEDUO if ceduo else GOV_FUSTAIA
        parcel_rows = particelle_df[(particelle_df[COL_COMPRESA] == compresa) &
                                    (particelle_df[COL_GOVERNO] == governo)]
        parcel_list = sorted(parcel_rows[COL_PARTICELLA].unique(), key=natsort_keygen())  # type: ignore[reportGeneralTypeIssues]
        if particelle:
            parcel_list = [p for p in parcel_list if p in particelle]

        output_parts = []
        for particella in parcel_list:
            expanded = modello_text.replace('@@compresa', compresa)
            expanded = expanded.replace('@@particella', str(particella))
            processed_part = re.sub(DIRECTIVE_PATTERN, process_directive, expanded)
            output_parts.append(processed_part)
        return '\n'.join(output_parts)

    def process_directive(match):
        directive = parse_template_directive(match.group(0))
        if not directive:
            return match.group(0)  # Return unchanged if parsing fails

        try:
            keyword = directive.keyword
            params = directive.params

            csv_unsupported = keyword.startswith('g')
            if format_type == Fmt.CSV and csv_unsupported:
                raise ValueError(
                    f"@@{keyword}: il formato CSV non supporta direttive grafiche (@@g*)")

            alberi_files = cast(list[str], params.get('alberi'))
            equazioni_files = cast(list[str], params.get(OPT_EQUAZIONI))

            if not alberi_files and keyword not in (Dir.PROP, Dir.PARCELS, Dir.COPPICE_SCHEDULE):
                raise ValueError(f"@@{keyword} richiede alberi=FILE")

            comprese = params.get('compresa', [])
            particelle = params.get('particella', [])
            generi = params.get('genere', [])

            if keyword == Dir.PROP:
                if len(comprese) != 1 or len(particelle) != 1 or len(params) != 2:
                    raise ValueError("@@prop richiede esattamente compresa=X e particella=Y")
                result = render_prop(particelle_df, comprese[0], particelle[0], formatter)
                return result.snippet

            if keyword == Dir.PROP_CEDUO:
                if len(comprese) != 1 or len(particelle) != 1:
                    raise ValueError("@@prop_ceduo richiede esattamente compresa=X e particella=Y")
                if not alberi_files:
                    raise ValueError("@@prop_ceduo richiede alberi=FILE")
                trees_df = load_trees(alberi_files, data_dir, ceduo=True)
                result = render_prop_coppice(
                    particelle_df, comprese[0], particelle[0], trees_df, formatter)
                return result.snippet

            if keyword == Dir.PARCELS:
                return render_particelle(comprese, particelle, particelle_df, params)

            if keyword == Dir.COPPICE_SCHEDULE:
                particelle_path = params.get(OPT_PARTICELLE)
                if not particelle_path:
                    raise ValueError("@@calendario_ceduo richiede 'particelle=FILE'")
                adiacenze_path = params.get(OPT_ADIACENZE)
                if not adiacenze_path:
                    raise ValueError("@@calendario_ceduo richiede 'adiacenze=FILE'")
                calendario_path = params.get(OPT_CALENDARIO)

                ceduo_parcels = load_coppice_parcels(data_dir / particelle_path)
                adjacencies = load_adjacencies(data_dir / adiacenze_path)
                ceduo_last = (
                    last_harvests_from_calendario(data_dir / calendario_path)
                    if calendario_path else {})

                anno_inizio = int(params.get(OPT_ANNO_INIZIO, 2027))
                anno_fine = int(params.get(OPT_ANNO_FINE, 2040))
                check_allowed_params(keyword, params,
                    {OPT_PARTICELLE: True, OPT_ADIACENZE: True,
                        OPT_CALENDARIO: True, OPT_ANNO_INIZIO: True,
                        OPT_ANNO_FINE: True})
                check_required_params(keyword, params,
                    [OPT_PARTICELLE, OPT_ADIACENZE])

                ceduo_events = schedule_coppice(
                    ceduo_parcels, adjacencies, ceduo_last,
                    (anno_inizio, anno_fine))
                result = render_coppice_schedule(ceduo_events, formatter)
                return result.snippet

            trees_df = load_trees(alberi_files, data_dir)
            data = parcel_data(alberi_files, trees_df, particelle_df, comprese, particelle, generi)

            match keyword:
                case Dir.VOLUME_TABLE:
                    options = {
                        OPT_PER_COMPRESA: _bool_opt(params, OPT_PER_COMPRESA),
                        OPT_PER_PARTICELLA: _bool_opt(params, OPT_PER_PARTICELLA),
                        OPT_PER_GENERE: _bool_opt(params, OPT_PER_GENERE),
                        OPT_INTERV_FIDUC: _bool_opt(params, OPT_INTERV_FIDUC, False),
                        OPT_SOLO_MATURE: _bool_opt(params, OPT_SOLO_MATURE, False),
                        OPT_STIME_TOTALI: _bool_opt(params, OPT_STIME_TOTALI),
                        OPT_TOTALI: _bool_opt(params, OPT_TOTALI, False),
                    }
                    check_allowed_params(keyword, params, options)
                    result = render_volume_table(data, formatter, **options)
                case Dir.HARVEST_TABLE:
                    if 'genere' in params:
                        raise ValueError("@@prelievi non supporta il parametro 'genere' "
                                         "(usa 'per_genere=si' per raggruppare per specie)")
                    options = {
                        OPT_PER_COMPRESA: _bool_opt(params, OPT_PER_COMPRESA),
                        OPT_PER_PARTICELLA: _bool_opt(params, OPT_PER_PARTICELLA),
                        OPT_PER_GENERE: _bool_opt(params, OPT_PER_GENERE, False),
                        OPT_COL_AREA_HA: _bool_opt(params, OPT_COL_AREA_HA),
                        OPT_COL_COMPARTO: _bool_opt(params, OPT_COL_COMPARTO),
                        OPT_COL_ETA: _bool_opt(params, OPT_COL_ETA),
                        OPT_COL_PP_MAX: _bool_opt(params, OPT_COL_PP_MAX),
                        OPT_COL_PRELIEVO_HA: _bool_opt(params, OPT_COL_PRELIEVO_HA),
                        OPT_COL_PRELIEVO: _bool_opt(params, OPT_COL_PRELIEVO),
                        OPT_COL_VOLUME_HA: _bool_opt(params, OPT_COL_VOLUME_HA, False),
                        OPT_COL_VOLUME_MATURE_HA: _bool_opt(params, OPT_COL_VOLUME_MATURE_HA),
                        OPT_COL_VOLUME_MATURE: _bool_opt(params, OPT_COL_VOLUME_MATURE),
                        OPT_COL_VOLUME: _bool_opt(params, OPT_COL_VOLUME, False),
                        OPT_TOTALI: _bool_opt(params, OPT_TOTALI, False),
                    }
                    check_allowed_params(keyword, params, options)
                    result = render_harvest_table(data, max_harvest,
                                              formatter, **options)
                case Dir.HARVEST_PLAN:
                    nonlocal harvest_plan_count
                    harvest_plan_count += 1
                    calendario_path = params.get(OPT_CALENDARIO)
                    past_harvests = (
                        read_past_harvests(data_dir / calendario_path)
                        if calendario_path else None)
                    anno_inizio = int(params.get(OPT_ANNO_INIZIO, 2026))
                    anno_fine = int(params.get(OPT_ANNO_FINE, 2040))
                    options = {
                        OPT_PER_COMPRESA: _bool_opt(params, OPT_PER_COMPRESA),
                        OPT_PER_PARTICELLA: _bool_opt(params, OPT_PER_PARTICELLA),
                        OPT_PER_GENERE: _bool_opt(params, OPT_PER_GENERE, False),
                        OPT_COL_COMPARTO: _bool_opt(params, OPT_COL_COMPARTO),
                        OPT_COL_ETA: _bool_opt(params, OPT_COL_ETA),
                        OPT_COL_PP_MAX: _bool_opt(params, OPT_COL_PP_MAX),
                        OPT_COL_PRIMA_DOPO: _bool_opt(params, OPT_COL_PRIMA_DOPO),
                        OPT_ANNO_FINE: anno_fine,
                        OPT_ANNO_INIZIO: anno_inizio,
                        OPT_INTERVALLO: int(params.get(OPT_INTERVALLO, 10)),
                        OPT_INTERVALLO_ANNO: parse_gap_overrides(
                            params.get(OPT_INTERVALLO_ANNO),
                            anno_inizio, anno_fine),
                        OPT_MORTALITA: float(params.get(OPT_MORTALITA, 0)),
                        OPT_PRUDENZA: float(params.get(OPT_PRUDENZA, 100)),
                        OPT_RIDUZIONE: float(params.get(OPT_RIDUZIONE, 100)),
                        OPT_TOTALI: _bool_opt(params, OPT_TOTALI, False),
                        OPT_VOLUME_OBIETTIVO: float(params[OPT_VOLUME_OBIETTIVO]),
                        OPT_ORDINE: params.get(OPT_ORDINE, ORDINE_VOL_HA),
                        OPT_PARTICELLE_MIN: int(params.get(OPT_PARTICELLE_MIN, 0)),
                    }
                    _VALID_ORDINE = {ORDINE_VOL_HA, ORDINE_VOL_TOT, ORDINE_DATA}
                    if options[OPT_ORDINE] not in _VALID_ORDINE:
                        raise ValueError(f"@@piano_di_taglio: ordine='{options[OPT_ORDINE]}' "
                                         f"non valido (valori ammessi: {', '.join(sorted(_VALID_ORDINE))})")
                    check_allowed_params(keyword, params,
                                         options | {OPT_CALENDARIO: True})
                    check_required_params(keyword, params,
                                          [OPT_VOLUME_OBIETTIVO])
                    if options[OPT_COL_PRIMA_DOPO] and not options[OPT_PER_PARTICELLA]:
                        raise ValueError("@@piano_di_taglio richiede 'per_particella=si' se si usa "
                                         "'col_prima_dopo=si', altrimenti i volumi prima/dopo "
                                         "non sono confrontabili")
                    volume_log = {} if log_simulazione else None
                    result = render_harvest_plan(data, past_harvests,
                                               max_harvest,
                                               formatter,
                                               volume_log=volume_log,
                                               **options)
                    if volume_log:
                        log_path = f'simulazione_pdt_{harvest_plan_count}.csv'
                        write_volume_log(volume_log, log_path)
                        print(f"  Log simulazione salvato in {log_path}")
                case Dir.PCT_GROWTH_TABLE:
                    options = {
                        OPT_PER_COMPRESA: _bool_opt(params, OPT_PER_COMPRESA, False),
                        OPT_PER_PARTICELLA: _bool_opt(params, OPT_PER_PARTICELLA, False),
                        OPT_STIME_TOTALI: _bool_opt(params, OPT_STIME_TOTALI),
                        OPT_TOTALI: _bool_opt(params, OPT_TOTALI, False),
                    }
                    check_allowed_params(keyword, params, options)
                    result = render_pct_growth_table(data, formatter, **options)
                case Dir.PCT_GROWTH_GRAPH:
                    options = {
                        OPT_PER_COMPRESA: _bool_opt(params, OPT_PER_COMPRESA, False),
                        OPT_PER_PARTICELLA: _bool_opt(params, OPT_PER_PARTICELLA, False),
                        OPT_METRICA: params.get(OPT_METRICA, 'ip'),
                        OPT_STILE: params.get(OPT_STILE),
                        OPT_STIME_TOTALI: _bool_opt(params, OPT_STIME_TOTALI),
                    }
                    check_allowed_params(keyword, params, options)
                    check_param_values(options, OPT_METRICA, ['ip', 'ic'], '@@grafico_incremento_percentuale')
                    filename = _build_graph_filename(comprese, particelle, generi, keyword)
                    result = render_pct_growth_graph(data, output_dir / filename,
                                              formatter, color_map, **options)
                case Dir.DIAMETER_CLASS_GRAPH:
                    options = {
                        OPT_METRICA: params.get(OPT_METRICA, 'alberi_ha'),
                        OPT_STILE: params.get(OPT_STILE),
                        OPT_STIME_TOTALI: _bool_opt(params, OPT_STIME_TOTALI),
                        OPT_X_MAX: int(params.get(OPT_X_MAX, 0)),
                        OPT_Y_MAX: int(params.get(OPT_Y_MAX, 0)),
                    }
                    check_allowed_params(keyword, params, options)
                    check_param_values(options, OPT_METRICA,
                        ['alberi_ha', 'G_ha', 'volume_ha',
                         'alberi_tot', 'G_tot', 'volume_tot', 'altezza'],
                        '@@grafico_classi_diametriche')
                    filename = _build_graph_filename(comprese, particelle, generi, keyword)
                    result = render_diameter_class_graph(data, output_dir / filename,
                                              formatter, color_map, **options)
                case Dir.DIAMETER_CLASS_TABLE:
                    options = {
                        OPT_METRICA: params.get(OPT_METRICA, 'alberi_ha'),
                        OPT_STIME_TOTALI: _bool_opt(params, OPT_STIME_TOTALI),
                    }
                    check_allowed_params(keyword, params, options)
                    check_param_values(options, OPT_METRICA,
                        ['alberi_ha', 'G_ha', 'volume_ha',
                         'alberi_tot', 'G_tot', 'volume_tot', 'altezza'],
                        '@@tabella_classi_diametriche')
                    result = render_diameter_class_table(data, formatter, **options)
                case Dir.HYPSOMETRIC_GRAPH:
                    options = {
                        OPT_EQUAZIONI: True,
                        OPT_STILE: params.get(OPT_STILE),
                        OPT_X_MAX: int(params.get(OPT_X_MAX, 0)),
                        OPT_Y_MAX: int(params.get(OPT_Y_MAX, 0)),
                    }
                    check_allowed_params(keyword, params, options)
                    check_required_params(keyword, params, [OPT_EQUAZIONI])
                    equations_df = load_csv(equazioni_files, data_dir)
                    filename = _build_graph_filename(comprese, particelle, generi, keyword)
                    result = render_hypsometric_graph(data, equations_df, output_dir / filename,
                                              formatter, color_map, **options)
                case _:
                    raise ValueError(f"Comando sconosciuto: {keyword}")

            return result.snippet

        except Exception as e:
            raise ValueError(f"ERRORE nella generazione di {directive.full_text}: {e}") from e

    match format_type:
        case Fmt.HTML:
            formatter = HTMLSnippetFormatter()
        case Fmt.CSV:
            formatter = CSVSnippetFormatter()
        case Fmt.TEX | Fmt.PDF:
            formatter = LaTeXSnippetFormatter()
        case _:
            raise ValueError(f"Formato non supportato: {format_type}")
    color_map = get_color_map()
    particelle_df = load_csv(parcel_file)

    # Find and replace all directives
    processed = re.sub(DIRECTIVE_PATTERN, process_directive, template_text)

    return processed


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_parcels(particelle_file: str) -> None:
    """List all (compresa, particella) pairs from particelle file."""
    df = load_csv(particelle_file)
    df = df.dropna(subset=[COL_COMPRESA, COL_PARTICELLA])
    for compresa in sorted(df[COL_COMPRESA].unique()):
        compresa_data = df[df[COL_COMPRESA] == compresa]
        particelle = sorted(compresa_data[COL_PARTICELLA].astype(str).unique(),  # type: ignore[reportGeneralTypeIssues]
                          key=natsort_keygen())
        for particella in particelle:
            print(f"  {compresa},{particella}")


# =============================================================================
# CLI COMMAND HANDLERS
# =============================================================================

def run_genera_equazioni(args):
    """Generate equations."""
    print(f"Generazione equazioni da fonte: {args.fonte_altezze}")
    print(f"Funzione: {args.funzione}")

    if args.fonte_altezze == 'ipsometro':
        equations_df = fit_curves_from_ipsometro(args.input, args.funzione)
    elif args.fonte_altezze == 'originali':
        equations_df = fit_curves_from_originali(args.input, args.funzione)
    elif args.fonte_altezze == 'tabelle':
        equations_df = fit_curves_from_tabelle(args.input, args.particelle, args.funzione)
    else:
        raise ValueError(f"Fonte altezze non supportata: {args.fonte_altezze}")

    if equations_df is not None:
        equations_df.to_csv(args.output, index=False, float_format="%.4f")
        print(f"Equazioni salvate in: {args.output}")
        print(f"Totale equazioni generate: {len(equations_df)}")
    else:
        print("ERRORE: Nessuna equazione generata (funzioni stub non implementate)")


def run_calcola_incrementi(args):
    """Calculate IP (incremento percentuale) for each tree."""
    print("Calcolo incrementi percentuali")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    trees_df = load_trees(args.input)
    trees_df['IP'] = trees_df[COL_PRESSLER] * 2 * trees_df[COL_L10_MM] / 100 / trees_df[COL_D_CM]
    trees_df.to_csv(args.output, index=False, float_format="%.6f")
    print(f"\nFile salvato: {args.output}")


def run_calcola_altezze_volumi(args):
    """Calculate heights and volumes in one pass."""
    print(f"Calcolo altezze con equazioni da: {args.equazioni}")
    print("Calcolo volumi con tavole del Tabacchi")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    trees_df = load_trees(args.input)
    equations_df = load_csv(args.equazioni)

    if args.coeff_pressler is not None:
        trees_df[COL_PRESSLER] = args.coeff_pressler
        trees_df['IP'] = trees_df[COL_PRESSLER] * 2 * trees_df[COL_L10_MM] / 100 / trees_df[COL_D_CM]
        print(f"Coefficiente di Pressler = {args.coeff_pressler} per tutti gli alberi")

    trees_df, updated, unchanged = compute_heights(trees_df, equations_df, verbose=True)
    trees_df = calculate_all_trees_volume(trees_df)
    print(f"\nCalcolo altezze e volumi: {updated} alberi aggiornati, {unchanged} immutati")

    trees_df.sort_values(
        by=[COL_COMPRESA, COL_PARTICELLA, COL_AREA_SAGGIO, 'n'],
        key=lambda col: col.map(natsort_keygen()) if col.name == COL_PARTICELLA else col,
        inplace=True)
    trees_df.to_csv(args.output, index=False, float_format="%.6f")
    print(f"\nFile salvato: {args.output}")


def run_report(args):
    """Generate report from template."""
    print(f"Generazione report formato: {args.formato}")
    print(f"Input: {args.input}")
    print(f"Cartella dati: {args.dati}")
    print(f"Cartella output: {args.output_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.input, 'r', encoding='utf-8') as f:
        template_text = f.read()

    processed = process_template(template_text, Path(args.dati), args.particelle,
                                 output_dir, args.formato, Path(args.input).parent,
                                 log_simulazione=args.log_simulazione)
    output_file = output_dir / Path(args.input).name
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(processed)
    if args.formato == Fmt.PDF:
        print("Esecuzione pdflatex no.1")
        subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', output_file.stem],
            cwd=output_dir,
            capture_output=True,
            check=True
        )
        print("Esecuzione biber")
        subprocess.run(
            ['biber', output_file.stem],
            cwd=output_dir,
            capture_output=True,
            check=True
        )
        for i in range(2):
            print(f"Esecuzione pdflatex no.{i+2}")
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', output_file.stem],
                cwd=output_dir,
                capture_output=True,
                check=True
            )

        print(f"Report generato: {output_file.with_suffix('.pdf')}")
    else:
        print(f"Report generato: {output_file}")


def run_lista_particelle(args):
    """List land parcels."""
    print("Particelle disponibili:")
    list_parcels(args.particelle)


# =============================================================================
# MAIN AND ARGUMENT PARSING
# =============================================================================

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Analisi Accrescimenti - Tool unificato',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modalità di utilizzo:

1. GENERA EQUAZIONI:
   ./pdg.py --genera-equazioni --funzione=log --fonte-altezze=ipsometro \\
            --input altezze.csv --output equations.csv

   ./pdg.py --genera-equazioni --funzione=log --fonte-altezze=tabelle \\
            --input alsometrie.csv --particelle particelle.csv --output equations.csv

2. CALCOLA ALTEZZE E VOLUMI:
   ./pdg.py --calcola-altezze-volumi --equazioni equations.csv \\
            --input alberi.csv --output alberi-calcolati.csv

3. GENERA REPORT:
   ./pdg.py --report --formato=html --dati csv/ --particelle particelle.csv \\
            --input template.html --output-dir report/
   (Directives specify alberi=file.csv and equazioni=file.csv)

4. LISTA PARTICELLE:
   ./pdg.py --lista-particelle --particelle particelle.csv
"""
    )

    # Mode selection (mutually exclusive)
    run_group = parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument('--genera-equazioni', action='store_true',
                           help='Genera equazioni di interpolazione')
    run_group.add_argument('--calcola-altezze-volumi', action='store_true',
                           help='Calcola altezze (equazioni) e volumi (Tabacchi)')
    run_group.add_argument('--calcola-incrementi', action='store_true',
                           help='Calcola incrementi percentuali (IP)')
    run_group.add_argument('--report', action='store_true',
                           help='Genera report da template')
    run_group.add_argument('--lista-particelle', action='store_true',
                           help='Lista particelle (compresa, particella)')

    # Common file arguments
    files_group = parser.add_argument_group('File di input/output')
    files_group.add_argument('--input',
                            help='File di input')
    files_group.add_argument('--output',
                            help='File di output')
    files_group.add_argument('--output-dir',
                            help='Directory di output (per report)')
    files_group.add_argument('--equazioni',
                            help='File CSV con equazioni (per --calcola-altezze-volumi)')
    files_group.add_argument('--dati',
                            help='Directory base per file dati (per --report)')
    files_group.add_argument('--particelle',
                            help='File CSV con dati particelle')

    # Specific options for --calcola-altezze-volumi
    av_group = parser.add_argument_group('Opzioni per --calcola-altezze-volumi')
    av_group.add_argument('--coeff-pressler', type=float, default=None,
                          help='Imposta coefficiente di Pressler per tutti gli alberi')

    # Specific options for --genera-equazioni
    eq_group = parser.add_argument_group('Opzioni per --genera-equazioni')
    eq_group.add_argument('--funzione', choices=['log', 'lin'], default='log',
                         help='Tipo di funzione (default: log)')
    eq_group.add_argument('--fonte-altezze',
                         choices=['ipsometro', 'originali', 'tabelle'],
                         help='Fonte dei dati di altezza')

    # Specific options for --report
    report_group = parser.add_argument_group('Opzioni per --report')
    report_group.add_argument('--formato', choices=[Fmt.CSV, Fmt.HTML, Fmt.TEX, Fmt.PDF], default=Fmt.PDF,
                             help='Formato output (default: pdf)')
    report_group.add_argument('--ometti-generi-sconosciuti', action='store_true',
                             help='Ometti dai grafici generi per cui non abbiamo equazioni')

    # Other options
    opt_group = parser.add_argument_group('Altre opzioni')
    opt_group.add_argument('--non-rigenerare-grafici', action='store_true', default=False,
                           help='Non rigenerare grafici esistenti (per --report)')
    opt_group.add_argument('--log-simulazione', action='store_true', default=False,
                           help='Scrivi CSV con volumi per particella per ogni @@piano_di_taglio')
    opt_group.add_argument('--separatore-decimale', choices=['punto', 'virgola'],
                           default='virgola',
                           help='Separatore decimale: punto (default) o virgola')

    args = parser.parse_args()

    if args.non_rigenerare_grafici:
        import pdg.core
        pdg.core.skip_graphs = True

    if args.separatore_decimale == 'virgola':
        set_decimal_comma(True)

    if args.genera_equazioni:
        if not args.fonte_altezze:
            parser.error('--genera-equazioni richiede --fonte-altezze')
        if not args.input:
            parser.error('--genera-equazioni richiede --input')
        if not args.output:
            parser.error('--genera-equazioni richiede --output')
        if args.fonte_altezze == 'tabelle' and not args.particelle:
            parser.error('--fonte-altezze=tabelle richiede --particelle')
        run_genera_equazioni(args)

    elif args.calcola_altezze_volumi:
        if not args.equazioni:
            parser.error('--calcola-altezze-volumi richiede --equazioni')
        if not args.input:
            parser.error('--calcola-altezze-volumi richiede --input')
        if not args.output:
            parser.error('--calcola-altezze-volumi richiede --output')
        run_calcola_altezze_volumi(args)

    elif args.calcola_incrementi:
        if not args.input:
            parser.error('--calcola-incrementi richiede --input')
        if not args.output:
            parser.error('--calcola-incrementi richiede --output')
        run_calcola_incrementi(args)

    elif args.report:
        if not args.dati:
            parser.error('--report richiede --dati')
        if not args.particelle:
            parser.error('--report richiede --particelle')
        if not args.input:
            parser.error('--report richiede --input')
        if not args.output_dir:
            parser.error('--report richiede --output-dir')
        run_report(args)

    elif args.lista_particelle:
        if not args.particelle:
            parser.error('--lista-particelle richiede --particelle')
        run_lista_particelle(args)


if __name__ == "__main__":
    main()
