"""Output formatting: SnippetFormatter ABC + HTML/LaTeX/CSV implementations, fmt_num."""

from abc import ABC, abstractmethod
import csv
from dataclasses import dataclass
import io
from pathlib import Path


@dataclass
class CurveInfo:
    """Metadata for one regression curve (used in @@grafico_classi_ipsometriche graph legends)."""
    genere: str
    equation: str
    r_squared: float
    n_points: int


# Option key used by format_image to extract the style parameter.
OPT_STILE = 'stile'

# =============================================================================
# NUMBER FORMATTING
# =============================================================================

_decimal_comma = False  # pylint: disable=invalid-name

def set_decimal_comma(enabled: bool):
    """Set the decimal separator to comma (Italian) or period (English)."""
    global _decimal_comma  # pylint: disable=global-statement
    _decimal_comma = enabled

def fmt_num(value: float, decimals: int) -> str:
    """Format a number with the configured decimal separator."""
    s = f"{value:.{decimals}f}"
    if _decimal_comma:
        s = s.replace('.', ',')
    return s


# =============================================================================
# RENDER RESULT AND COLUMN SPEC
# =============================================================================

@dataclass
class RenderResult:
    """Result of a render_* function: snippet for template insertion, optional file path."""
    snippet: str
    filepath: Path | None = None

@dataclass
class ColSpec:
    """Column specification for table rendering."""
    # Column title
    title: str
    # Alignment: 'l', 'r', or 'c'
    align: str
    # Format spec: column name to display as .1f, or custom format function
    format: str | object  # str or Callable
    # Total spec: column name to sum, or callable(df) -> str, or None for no total
    total: str | object | None  # str, Callable, or None
    # True if column should be rendered
    enabled: bool


# =============================================================================
# SNIPPET FORMATTERS
# =============================================================================

class SnippetFormatter(ABC):
    """Formats individual components (images, metadata) for template insertion."""

    @abstractmethod
    def format_image(self, filepath: Path, options: dict | None = None) -> str:
        """Format image reference for this format."""

    @abstractmethod
    def format_metadata(self, data, curve_info: list[CurveInfo] | None = None) -> str:
        """Format metadata block for this format."""

    @abstractmethod
    def format_table(self, headers: list[tuple[str, str]], rows: list[list[str]],
                     row_groups: list[int] | None = None) -> str:
        """Format a data table for this format.

        Args:
            headers: Column headers as (title, alignment) tuples
            rows: Data rows (each row is a list of strings)
            row_groups: Row indices where a new group starts (for visual separation).
                        The first group (index 0) gets no separator; subsequent groups
                        get a visual break (hline in LaTeX, CSS class in HTML).
        Returns:
            Formatted table snippet
        """

    @abstractmethod
    def format_prop(self, short_fields: list[tuple[str, str]],
                    paragraph_fields: list[tuple[str, str]]) -> str:
        """Format parcel properties for this format."""


class HTMLSnippetFormatter(SnippetFormatter):
    """HTML snippet formatter."""

    def format_image(self, filepath: Path, options: dict | None = None) -> str:
        cls = options[OPT_STILE] if options and options[OPT_STILE] else 'graph-image'
        return f'<img src="{filepath.name}" class="{cls}">'

    def format_metadata(self, data, curve_info: list[CurveInfo] | None = None) -> str:
        """Format metadata as HTML."""
        html = '<div class="graph-details">\n'
        html += f'<p><strong>Comprese:</strong> {data.regions}</p>\n'
        html += f'<p><strong>Generi:</strong> {data.species}</p>\n'
        html += f'<p><strong>Alberi campionati:</strong> {data.trees.shape[0]:d}</p>\n'
        if curve_info:
            i = 'i' if len(curve_info) > 1 else 'e'
            html += f'<br><p><strong>Equazion{i} interpolant{i}:</strong></p>\n'
            for curve in curve_info:
                html += (f'<p>{curve.genere}: {curve.equation} '
                         f'(R² = {fmt_num(curve.r_squared, 2)}, n = {curve.n_points})</p>\n')
        html += '</div>\n'
        return html

    def format_table(self, headers: list[tuple[str, str]], rows: list[list[str]],
                     row_groups: list[int] | None = None) -> str:
        """Format table as HTML.
        Headers is a list of tuples (header, justification).
        Justification is 'l' for left, 'r' for right, 'c' for center.
        """
        justify_style = {'l': 'col_left', 'r': 'col_right', 'c': 'col_center'}
        justify = [justify_style[h[1]] for h in headers]
        # Group breaks that need a separator (skip the first group).
        group_breaks = set(row_groups[1:]) if row_groups else set()
        html = '<table class="volume-table">\n'
        html += '  <thead>\n    <tr>\n'
        for header, j in zip([h[0].replace('\n', '<br>') for h in headers], justify):
            html += f'      <th class="{j}">{header}</th>\n'
        html += '    </tr>\n  </thead>\n'
        html += '  <tbody>\n'
        for i, row in enumerate(rows):
            cls = ' class="group-first"' if i in group_breaks else ''
            html += f'    <tr{cls}>\n'
            for cell, j in zip(row, justify):
                html += f'      <td class="{j}">{cell}</td>\n'
            html += '    </tr>\n'
        html += '  </tbody>\n</table>\n'
        return html

    def format_prop(self, short_fields: list[tuple[str, str]],
                    paragraph_fields: list[tuple[str, str]]) -> str:
        """Format parcel properties as HTML."""
        html = '<div class="parcel-props">\n'
        html += '<p class="props-inline">'
        html += ' · '.join(f'<strong>{label}:</strong> {value}' for label, value in short_fields)
        html += '</p>\n'
        for label, value in paragraph_fields:
            html += f'<p><strong>{label}:</strong> {value}</p>\n'
        html += '</div>\n'
        return html


class LaTeXSnippetFormatter(SnippetFormatter):
    """LaTeX snippet formatter."""

    def format_image(self, filepath: Path, options: dict | None = None) -> str:
        fmt = options[OPT_STILE] if options and options[OPT_STILE] else 'width=0.5\\textwidth'
        latex = '\\begin{center}\n'
        latex += f'  \\includegraphics[{fmt}]{{{filepath.name}}}\n'
        latex += '\\end{center}\n'
        return latex

    def format_metadata(self, data, curve_info: list[CurveInfo] | None = None) -> str:
        """Format metadata as LaTeX."""
        if not curve_info:
            return ""
        latex = '\\begin{quote}\\small\n'
        i = 'i' if len(curve_info) > 1 else 'e'
        latex += f'\n\\textbf{{Equazion{i} interpolant{i}:}}\\\\\n'
        for curve in curve_info:
            eq = curve.equation.replace('*', r'\times ')
            eq = eq.replace('ln', r'\ln')
            eq = eq.replace(',', '{,}')
            latex += (f"{curve.genere}: ${eq}$ ($R^2$ = {fmt_num(curve.r_squared, 2)}, "
                        f"$n$ = {curve.n_points})\\\\\n")
        latex += '\\end{quote}\n'
        return latex

    def format_table(self, headers: list[tuple[str, str]], rows: list[list[str]],
                     row_groups: list[int] | None = None) -> str:
        """Format table as LaTeX using longtable for page breaks.
           Headers is a list of tuples (header, justification).
           Justification is 'l' for left, 'r' for right, 'c' for center.
        """
        just = [h[1] for h in headers]
        # Group breaks that need a separator (skip the first group).
        group_breaks = set(row_groups[1:]) if row_groups else set()

        def _latex_header(title, align):
            if '\n' in title:
                lines = title.split('\n')
                return '\\shortstack[' + align + ']{' + '\\\\'.join(lines) + '}'
            return title

        header_titles = [_latex_header(h[0], h[1]) for h in headers]

        # Use longtable instead of tabular to allow page breaks
        latex = f'\\begin{{longtable}}{{ {"".join(just)} }}\n'
        latex += '\\hline\n'
        latex += ' & '.join(header_titles) + ' \\\\\n'
        latex += '\\hline\n'
        latex += '\\endfirsthead\n'  # End of first page header
        latex += '\\multicolumn{' + str(len(headers)) + '}{c}'
        latex += '{\\textit{(continua dalla pagina precedente)}} \\\\\n'
        latex += '\\hline\n'
        latex += ' & '.join(header_titles) + ' \\\\\n'
        latex += '\\hline\n'
        latex += '\\endhead\n'  # Header for subsequent pages
        latex += '\\hline\n'
        latex += '\\multicolumn{' + str(len(headers)) + '}{r}'
        latex += '{\\textit{(continua alla pagina successiva)}} \\\\\n'
        latex += '\\endfoot\n'  # Footer for all pages except last
        latex += '\\hline\n'
        latex += '\\endlastfoot\n'  # Footer for last page
        for i, row in enumerate(rows):
            if i in group_breaks:
                latex += '\\hline\n'
            latex += ' & '.join(row) + ' \\\\\n'
        latex += '\\end{longtable}\n'
        return latex

    def format_prop(self, short_fields: list[tuple[str, str]],
                    paragraph_fields: list[tuple[str, str]]) -> str:
        """Format parcel properties as LaTeX."""
        formatted = [f'\\textbf{{{label}:}} {value}' for label, value in short_fields]
        lines = [' $\\cdot$ '.join(formatted[i:i+2]) for i in range(0, len(formatted), 2)]
        latex = '\\noindent ' + ' \\\\\n'.join(lines) + '\n\n'
        for label, value in paragraph_fields:
            latex += f'\\noindent\\textbf{{{label}:}} {value}\n\n'
        return latex


class CSVSnippetFormatter(SnippetFormatter):
    """CSV snippet formatter for table-only output."""

    def format_image(self, filepath: Path, options: dict | None = None) -> str:
        raise NotImplementedError("Formato CSV non supporta immagini (direttive @@g*)")

    def format_metadata(self, data, curve_info: list[CurveInfo] | None = None) -> str:
        raise NotImplementedError("Formato CSV non supporta metadati")

    def format_table(self, headers: list[tuple[str, str]], rows: list[list[str]],
                     row_groups: list[int] | None = None) -> str:
        """Format table as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([h[0].replace('\n', ' ') for h in headers])
        writer.writerows(rows)
        return output.getvalue()

    def format_prop(self, short_fields: list[tuple[str, str]],
                    paragraph_fields: list[tuple[str, str]]) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([label for label, _ in short_fields + paragraph_fields])
        writer.writerow([value for _, value in short_fields + paragraph_fields])
        return output.getvalue()
