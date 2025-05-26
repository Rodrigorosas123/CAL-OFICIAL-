import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
import io
import sys
import os
import contextlib
import threading
import statsmodels.api as sm
from scipy import stats
import pickle # Para guardar/cargar sesiones
from datetime import datetime # Para timestamp en exportación

# Importaciones específicas para la creación del ícono (Pillow)
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("ADVERTENCIA: La librería Pillow (PIL) no se encontró. No se podrá crear el ícono de la aplicación. Instálala con 'pip install Pillow'")
    class Image:
        @staticmethod
        def new(*args, **kwargs): return None
    class ImageDraw:
        @staticmethod
        def Draw(*args, **kwargs): return None
    class ImageFont:
        @staticmethod
        def truetype(*args, **kwargs): return None
        @staticmethod
        def load_default(): return None

# Importación para PDF
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.units import inch
except ImportError:
    print("ADVERTENCIA: La librería ReportLab no se encontró. No se podrá exportar a PDF. Instálala con 'pip install reportlab'")
    # Define clases dummy para evitar errores si no está instalada
    class SimpleDocTemplate:
        def __init__(self, *args, **kwargs): pass
        def build(self, *args, **kwargs): print("ReportLab no está instalado. No se puede generar PDF.")
    class Paragraph:
        def __init__(self, *args, **kwargs): pass
    class Spacer:
        def __init__(self, *args, **kwargs): pass
    class RLImage:
        def __init__(self, *args, **kwargs): pass
    class PageBreak:
        pass
    class getSampleStyleSheet:
        def __init__(self): pass
        def __getitem__(self, key): return ParagraphStyle(name=key)
    class ParagraphStyle:
        def __init__(self, *args, **kwargs): pass


# --- CONFIGURACIÓN DE ESTILOS Y TEMAS ---
def apply_custom_styles(style_obj):
    """Aplica estilos ttk personalizados para una apariencia profesional."""
    COLORS = {
        "primary": "#2C3E50",   # Gris Azul Oscuro
        "accent": "#3498DB",    # Azul Cielo
        "light_bg": "#ECF0F1",  # Gris Claro
        "medium_bg": "#BDC3C7", # Gris Medio
        "dark_text": "#2C3E50", # Texto oscuro
        "light_text": "#FFFFFF",# Texto blanco
        "border": "#AAB7B8",    # Borde sutil
        "success": "#2ECC71",   # Verde
        "warning": "#F39C12",   # Naranja
        "error": "#E74C4C",     # Rojo
        "plot_bg_color": "#DDEBF7" # Fondo del área de gráficos
    }

    style_obj.theme_use("clam")

    style_obj.configure("TFrame", background=COLORS["light_bg"])
    style_obj.configure("TLabel", background=COLORS["light_bg"], foreground=COLORS["dark_text"], font=("Segoe UI", 10))
    style_obj.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6, background=COLORS["primary"], foreground=COLORS["light_text"])
    style_obj.map("TButton", background=[('active', COLORS["accent"])])
    
    style_obj.configure("TNotebook", background=COLORS["medium_bg"], borderwidth=0)
    style_obj.configure("TNotebook.Tab", background=COLORS["primary"], foreground=COLORS["light_text"], padding=[10, 5], font=("Segoe UI", 10, "bold"))
    style_obj.map("TNotebook.Tab", background=[('selected', COLORS["accent"])], foreground=[('selected', COLORS["light_text"])])

    style_obj.configure("Treeview", font=("Segoe UI", 9), rowheight=24, background=COLORS["light_text"], foreground=COLORS["dark_text"], fieldbackground=COLORS["light_text"])
    style_obj.map("Treeview", background=[('selected', COLORS["accent"])], foreground=[('selected', COLORS["light_text"])])
    style_obj.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), background=COLORS["medium_bg"], foreground=COLORS["dark_text"])
    style_obj.map("Treeview.Heading", background=[('active', COLORS["border"])])

    style_obj.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground=COLORS["primary"], background=COLORS["light_bg"])
    style_obj.configure("SubHeader.TLabel", font=("Segoe UI", 12, "bold"), foreground=COLORS["accent"], background=COLORS["light_bg"])
    style_obj.configure("Info.TLabel", font=("Segoe UI", 9, "italic"), foreground="#6C7A89", background=COLORS["light_bg"])
    style_obj.configure("PlotFrame.TFrame", background=COLORS["plot_bg_color"]) 
    style_obj.configure("TLabelframe", background=COLORS["light_bg"], foreground=COLORS["primary"], font=("Segoe UI", 11, "bold"))
    style_obj.configure("TLabelframe.Label", background=COLORS["light_bg"], foreground=COLORS["primary"])


# --- data_handler.py ---
class DataHandler:
    """Maneja la carga, almacenamiento y operaciones básicas de previsualización de datos."""
    def __init__(self):
        self.df = None
        self.original_df = None # Para operaciones de limpieza
        self.filepath = None
        self.analysis_history = [] # Para almacenar resultados de análisis para el meta-análisis

    def load_data(self, filepath: str) -> bool:
        """Carga datos desde una ruta de archivo. Retorna True si éxito, False si error."""
        try:
            if filepath.lower().endswith('.csv'):
                df_loaded = pd.read_csv(filepath)
            elif filepath.lower().endswith(('.xls', '.xlsx')):
                df_loaded = pd.read_excel(filepath)
            else:
                raise ValueError("Formato de archivo no soportado. Por favor, use .csv o .xlsx.")
            
            # Intentar inferir tipos de datos para optimizar y convertir a categóricas
            for col in df_loaded.columns:
                if df_loaded[col].nunique() / len(df_loaded) < 0.1 and df_loaded[col].dtype == 'object':
                    df_loaded[col] = df_loaded[col].astype('category')
            
            self.df = df_loaded.copy()
            self.original_df = df_loaded.copy() # Mantener una copia original
            self.filepath = filepath
            self.analysis_history = [] # Reiniciar historial al cargar nuevos datos
            return True
        except FileNotFoundError:
            raise FileNotFoundError(f"El archivo no se encontró: {filepath}")
        except pd.errors.EmptyDataError:
            raise ValueError("El archivo está vacío.")
        except pd.errors.ParserError:
            raise ValueError("No se pudo parsear el archivo. Verifique el formato y el delimitador (para CSV).")
        except Exception as e:
            raise Exception(f"Ocurrió un error inesperado al cargar el archivo: {e}")

    def _reset_data(self):
        """Reinicia el DataFrame interno y la ruta del archivo."""
        self.df = None
        self.original_df = None
        self.filepath = None
        self.analysis_history = []

    def get_dataframe(self) -> pd.DataFrame | None:
        """Retorna el DataFrame cargado (o el actual después de limpieza)."""
        return self.df

    def get_original_dataframe(self) -> pd.DataFrame | None:
        """Retorna el DataFrame original sin modificaciones de limpieza."""
        return self.original_df

    def get_column_names(self, df: pd.DataFrame | None = None) -> list[str]:
        """Retorna una lista de nombres de columna del DF actual o uno dado."""
        target_df = df if df is not None else self.df
        return target_df.columns.tolist() if target_df is not None else []

    def get_numeric_columns(self, df: pd.DataFrame | None = None) -> list[str]:
        """Retorna una lista de nombres de columnas numéricas del DF actual o uno dado."""
        target_df = df if df is not None else self.df
        if target_df is None: return []
        return target_df.select_dtypes(include=np.number).columns.tolist()

    def get_categorical_columns(self, df: pd.DataFrame | None = None) -> list[str]:
        """Retorna una lista de nombres de columnas categóricas (object/string/category) del DF actual o uno dado."""
        target_df = df if df is not None else self.df
        if target_df is None: return []
        return target_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    def get_column_data_types(self) -> pd.Series | None:
        """Retorna los tipos de datos de las columnas."""
        return self.df.dtypes if self.df is not None else None

    def get_missing_values_info(self) -> pd.DataFrame | None:
        """Retorna un DataFrame con el conteo y porcentaje de valores faltantes."""
        if self.df is None: return None
        missing_count = self.df.isnull().sum()
        missing_percent = (self.df.isnull().sum() / len(self.df)) * 100
        missing_df = pd.DataFrame({'Missing Count': missing_count, 'Missing Percent': missing_percent})
        return missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)

    def handle_missing_values(self, strategy: str, columns: list[str] = None, value=None):
        """Aplica una estrategia de manejo de valores faltantes al DataFrame."""
        if self.df is None: raise ValueError("No hay datos cargados para manejar valores faltantes.")
        
        cols_to_process = columns if columns else self.df.columns
        
        for col in cols_to_process:
            if col not in self.df.columns:
                print(f"Advertencia: La columna '{col}' no existe en el DataFrame actual. Saltando.")
                continue

            try:
                if strategy == "drop_rows":
                    # Drop rows only if *any* of the specified columns have NaN
                    initial_rows = len(self.df)
                    self.df.dropna(subset=cols_to_process, inplace=True)
                    rows_removed = initial_rows - len(self.df)
                    self.add_analysis_event("Limpieza de Datos", f"Se eliminaron {rows_removed} filas con valores nulos en columnas: {', '.join(cols_to_process)}.")
                    break # Si se eliminan filas, el bucle por columna no tiene sentido, ya se aplicó a todo el subset.
                elif strategy == "drop_columns":
                    self.df.drop(columns=col, inplace=True)
                    self.add_analysis_event("Limpieza de Datos", f"Se eliminó la columna '{col}' debido a valores nulos.")
                elif strategy == "impute_mean" and pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                    self.add_analysis_event("Limpieza de Datos", f"Valores nulos en '{col}' imputados con la media.")
                elif strategy == "impute_median" and pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                    self.add_analysis_event("Limpieza de Datos", f"Valores nulos en '{col}' imputados con la mediana.")
                elif strategy == "impute_mode":
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    self.add_analysis_event("Limpieza de Datos", f"Valores nulos en '{col}' imputados con la moda.")
                elif strategy == "impute_value":
                    self.df[col].fillna(value, inplace=True)
                    self.add_analysis_event("Limpieza de Datos", f"Valores nulos en '{col}' imputados con el valor '{value}'.")
                else:
                    if strategy != "drop_columns": # drop_columns puede aplicarse a no numéricas
                        print(f"Advertencia: Estrategia '{strategy}' no aplicable a columna '{col}' (tipo/estrategia incompatible).")
            except Exception as e:
                raise Exception(f"Error al manejar valores faltantes en columna '{col}' con estrategia '{strategy}': {e}")
        
        if strategy == "drop_rows":
            return f"Filas con nulos en '{', '.join(cols_to_process)}' eliminadas."
        elif strategy == "drop_columns":
            return f"Columnas '{', '.join(cols_to_process)}' (con nulos) eliminadas."
        elif strategy.startswith("impute_"):
            return f"Valores nulos en '{', '.join(cols_to_process)}' imputados con {strategy.replace('impute_', '')}{f'={value}' if strategy == 'impute_value' else ''}."
        return "Operación de manejo de nulos completada."

    def add_analysis_event(self, event_type: str, description: str, result_summary: str = "", timestamp: datetime = None):
        """Añade un evento al historial de análisis para el meta-análisis."""
        if timestamp is None:
            timestamp = datetime.now()
        self.analysis_history.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "type": event_type,
            "description": description,
            "result_summary": result_summary
        })

    def get_analysis_history(self) -> list[dict]:
        """Retorna el historial de eventos de análisis."""
        return self.analysis_history

# --- stats_analyzer.py ---
class StatsAnalyzer:
    """Realiza análisis estadísticos básicos y avanzados."""
    def __init__(self, data_handler: DataHandler):
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("deep")
        self.data_handler = data_handler # Referencia al data_handler para registrar eventos

    # --- Descriptivos ---
    def get_descriptive_stats(self, df: pd.DataFrame, column: str) -> str:
        """Calcula y retorna estadísticas descriptivas para una columna."""
        if df is None or column not in df.columns:
            return "Error: DataFrame no cargado o columna no válida."
        if df[column].isnull().all():
            return f"Error: La columna '{column}' contiene solo valores nulos. No se pueden calcular estadísticas."

        output_buffer = io.StringIO()
        with contextlib.redirect_stdout(output_buffer):
            print(f"--- Estadísticas Descriptivas para '{column}' ---\n")
            if pd.api.types.is_numeric_dtype(df[column]):
                series = df[column].dropna()
                if series.empty: return f"Error: La columna '{column}' no tiene valores numéricos válidos."
                desc_stats = series.describe()
                print(desc_stats.to_string())
                print("\n--- Medidas de Posición (Cuartiles) ---")
                print(f"Q1 (25%): {series.quantile(0.25):.2f}")
                print(f"Mediana (50%): {series.median():.2f}")
                print(f"Q3 (75%): {series.quantile(0.75):.2f}")
                iqr_val = series.quantile(0.75) - series.quantile(0.25)
                print(f"IQR (Q3-Q1): {iqr_val:.2f}")
                skew_val = series.skew()
                kurt_val = series.kurt()
                print(f"Asimetría (Skewness): {skew_val:.2f}")
                print(f"Curtosis (Kurtosis): {kurt_val:.2f}")

                # Registrar para meta-análisis
                self.data_handler.add_analysis_event(
                    "Estadísticas Descriptivas",
                    f"Cálculo de descriptivos para la columna '{column}'.",
                    f"Media: {desc_stats.loc['mean']:.2f}, Mediana: {series.median():.2f}, Desv. Std.: {desc_stats.loc['std']:.2f}, Sesgo: {skew_val:.2f}, Curtosis: {kurt_val:.2f}"
                )
            else:
                series = df[column].dropna()
                if series.empty: return f"Error: La columna '{column}' no tiene valores categóricos válidos."
                value_counts = series.value_counts()
                print(value_counts.to_string())
                print(f"\nNúmero de valores únicos: {series.nunique()}")
                mode_val = series.mode().iloc[0] if not series.mode().empty else "N/A"
                print(f"Moda: {mode_val}")
                # Registrar para meta-análisis
                self.data_handler.add_analysis_event(
                    "Estadísticas Descriptivas",
                    f"Cálculo de descriptivos para la columna categórica '{column}'.",
                    f"Categorías únicas: {series.nunique()}, Moda: {mode_val}, Top 3: {value_counts.head(3).to_dict()}"
                )
        return output_buffer.getvalue()

    # --- Gráficos ---
    def plot_histogram(self, df: pd.DataFrame, column: str) -> tuple[plt.Figure | None, str | None, str | None]:
        """
        Genera un histograma con KDE para una columna numérica y proporciona su interpretación.
        Retorna la figura, un mensaje de error (si lo hay) y la interpretación del gráfico.
        """
        if df is None or column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            return None, "Selecciona una columna numérica válida para el histograma.", None
        series = df[column].dropna()
        if series.empty: return None, f"La columna '{column}' no tiene valores numéricos válidos para graficar.", None
        
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(series, kde=True, ax=ax, color='#3498DB', edgecolor='white', linewidth=0.5)
        ax.set_title(f'Distribución de {column}', fontsize=14, fontweight='bold')
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel('Frecuencia', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()

        # --- Interpretación del Histograma ---
        interpretation = (
            f"El **histograma de '{column}'** muestra la distribución de los valores. "
            f"La forma de la distribución (simétrica, sesgada a la izquierda/derecha), "
            f"el número de picos (unimodal, bimodal) y la dispersión (ancha, estrecha) "
            f"son clave para entender el comportamiento de esta variable. "
            f"La curva **KDE (Estimación de Densidad del Kernel)** suaviza el histograma para dar una mejor idea de la forma subyacente de la distribución."
        )
        
        skewness = series.skew()
        kurtosis = series.kurt()

        if skewness > 0.5:
            interpretation += f"\n\n**Sesgo:** Indica un **sesgo positivo (a la derecha)** pronunciado ({skewness:.2f}), lo que significa que la cola de la distribución es más larga hacia valores altos, y la mayoría de los datos se agrupan en la parte inferior."
        elif skewness < -0.5:
            interpretation += f"\n\n**Sesgo:** Indica un **sesgo negativo (a la izquierda)** pronunciado ({skewness:.2f}), lo que significa que la cola es más larga hacia valores bajos, y la mayoría de los datos se agrupan en la parte superior."
        else:
            interpretation += f"\n\n**Sesgo:** La distribución parece ser relativamente **simétrica** ({skewness:.2f}), con la mayoría de los datos agrupados alrededor de la media."
        
        if kurtosis > 0.5:
            interpretation += f"\n**Curtosis:** Presenta **leptocurtosis** ({kurtosis:.2f}), lo que implica que tiene picos más altos y colas más pesadas que una distribución normal (más valores en los extremos y cerca de la media). Esto sugiere una mayor concentración de datos en el centro y más valores atípicos."
        elif kurtosis < -0.5:
            interpretation += f"\n**Curtosis:** Presenta **platocurtosis** ({kurtosis:.2f}), lo que implica que tiene un pico más plano y colas más ligeras que una distribución normal (menos valores en los extremos y cerca de la media). Esto sugiere una distribución más dispersa con menos concentración en el centro y menos valores atípicos."
        else:
            interpretation += f"\n**Curtosis:** Su curtosis es similar a una distribución normal (mesocúrtica) ({kurtosis:.2f}), indicando una dispersión y concentración de datos moderada."

        self.data_handler.add_analysis_event("Visualización", f"Histograma de '{column}'.", f"Sesgo: {skewness:.2f}, Curtosis: {kurtosis:.2f}. {interpretation.split('.')[0]}.")
        return fig, None, interpretation

    def plot_boxplot(self, df: pd.DataFrame, column: str) -> tuple[plt.Figure | None, str | None, str | None]:
        """
        Genera un boxplot para una columna numérica y proporciona su interpretación.
        Retorna la figura, un mensaje de error (si lo hay) y la interpretación del gráfico.
        """
        if df is None or column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            return None, "Selecciona una columna numérica válida para el boxplot.", None
        series = df[column].dropna()
        if series.empty: return None, f"La columna '{column}' no tiene valores numéricos válidos para graficar.", None

        fig, ax = plt.subplots(figsize=(5, 6))
        sns.boxplot(y=series, ax=ax, color='#2ECC71', width=0.4, fliersize=5)
        ax.set_title(f'Boxplot de {column}', fontsize=14, fontweight='bold')
        ax.set_ylabel(column, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()

        # --- Interpretación del Boxplot ---
        q1 = series.quantile(0.25)
        median = series.median()
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]

        interpretation = (
            f"El **boxplot de '{column}'** ilustra la distribución de los datos a través de sus cuartiles. "
            f"La **caja central** representa el **50% de los datos** (desde el Cuartil 1: {q1:.2f} al Cuartil 3: {q3:.2f}). "
            f"La línea dentro de la caja es la **mediana** ({median:.2f}), indicando el punto medio de los datos. "
            f"Los 'bigotes' se extienden hasta los valores mínimo y máximo dentro de 1.5 veces el Rango Intercuartílico (IQR: {iqr:.2f}) desde los cuartiles. "
            f"Los puntos individuales fuera de los bigotes son considerados **valores atípicos (outliers)**, lo que indica observaciones inusuales."
        )
        
        if median > series.mean():
            interpretation += f"\n\n- La mediana está por encima de la media ({series.mean():.2f}), sugiriendo un ligero sesgo a la izquierda (más valores altos)."
        elif median < series.mean():
            interpretation += f"\n\n- La mediana está por debajo de la media ({series.mean():.2f}), sugiriendo un ligero sesgo a la derecha (más valores bajos)."
        else:
            interpretation += f"\n\n- La mediana y la media están muy cerca, lo que sugiere una distribución más simétrica."

        if not outliers.empty:
            interpretation += f"\n- Se observan **{len(outliers)} valores atípicos** en los datos. Esto merece una investigación para entender su origen (errores de datos, eventos inusuales, etc.). Los valores atípicos pueden influir significativamente en el promedio y la varianza."
        else:
            interpretation += f"\n- No se observan valores atípicos significativos, lo que indica una distribución más contenida y predecible."

        self.data_handler.add_analysis_event("Visualización", f"Boxplot de '{column}'.", f"Mediana: {median:.2f}, IQR: {iqr:.2f}, Outliers: {len(outliers)}.")
        return fig, None, interpretation

    def plot_scatterplot(self, df: pd.DataFrame, x_column: str, y_column: str) -> tuple[plt.Figure | None, str | None, str | None]:
        """
        Genera un scatterplot para dos columnas numéricas y proporciona su interpretación.
        Retorna la figura, un mensaje de error (si lo hay) y la interpretación del gráfico.
        """
        if df is None or not (x_column in df.columns and y_column in df.columns):
            return None, "Selecciona columnas válidas para el scatterplot.", None
        if not (pd.api.types.is_numeric_dtype(df[x_column]) and pd.api.types.is_numeric_dtype(df[y_column])):
            return None, "Ambas columnas deben ser numéricas para el scatterplot.", None
        
        temp_df = df[[x_column, y_column]].dropna()
        if temp_df.empty: return None, "No hay pares de valores válidos en las columnas seleccionadas para el scatterplot.", None

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.scatterplot(x=temp_df[x_column], y=temp_df[y_column], ax=ax, alpha=0.7, color='#9B59B6')
        
        correlation = temp_df[x_column].corr(temp_df[y_column])
        if abs(correlation) > 0.3:
            sns.regplot(x=temp_df[x_column], y=temp_df[y_column], ax=ax, scatter=False, color='#E74C4C', line_kws={'linestyle':'--', 'linewidth':1.5})
            
        ax.set_title(f'Diagrama de Dispersión: {x_column} vs {y_column}', fontsize=14, fontweight='bold')
        ax.set_xlabel(x_column, fontsize=12)
        ax.set_ylabel(y_column, fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        fig.tight_layout()

        # --- Interpretación del Scatterplot ---
        interpretation = (
            f"El **diagrama de dispersión** muestra la relación entre '{x_column}' y '{y_column}'. "
            f"Cada punto representa una observación con sus valores para ambas variables. "
            f"Este gráfico es ideal para identificar si existe una **correlación**, su dirección (positiva/negativa) y fuerza (fuerte/débil)."
        )
        
        if correlation > 0.7:
            interpretation += f"\n\n**Correlación:** Existe una **fuerte correlación positiva** ({correlation:.2f}). A medida que '{x_column}' aumenta, '{y_column}' también tiende a aumentar de manera consistente. Esto sugiere que ambas variables se mueven en la misma dirección de forma predecible."
        elif correlation > 0.3:
            interpretation += f"\n\n**Correlación:** Existe una **correlación positiva moderada** ({correlation:.2f}). Hay una tendencia a que '{y_column}' aumente con '{x_column}', aunque con más dispersión. La relación es discernible pero no tan estricta."
        elif correlation < -0.7:
            interpretation += f"\n\n**Correlación:** Existe una **fuerte correlación negativa** ({correlation:.2f}). A medida que '{x_column}' aumenta, '{y_column}' tiende a disminuir de manera consistente. Esto sugiere que las variables se mueven en direcciones opuestas de forma predecible."
        elif correlation < -0.3:
            interpretation += f"\n\n**Correlación:** Existe una **correlación negativa moderada** ({correlation:.2f}). Hay una tendencia a que '{y_column}' disminuya con '{x_column}', aunque con más dispersión. La relación es discernible pero con variabilidad."
        else:
            interpretation += f"\n\n**Correlación:** La correlación lineal entre '{x_column}' y '{y_column}' es **débil o inexistente** ({correlation:.2f}). No parece haber una relación lineal clara entre estas variables. Esto no descarta una relación no lineal."
        
        interpretation += f"\n\nBusca también **patrones no lineales** (curvas, etc.) o **grupos de puntos (clusters)** que puedan indicar subpoblaciones o relaciones más complejas no capturadas por la correlación lineal."
        self.data_handler.add_analysis_event("Visualización", f"Diagrama de Dispersión: '{x_column}' vs '{y_column}'.", f"Correlación: {correlation:.2f}.")
        return fig, None, interpretation

    def plot_grouped_boxplot(self, df: pd.DataFrame, numeric_col: str, category_col: str) -> tuple[plt.Figure | None, str | None, str | None]:
        """
        Genera un boxplot agrupado por una columna categórica y proporciona su interpretación.
        Retorna la figura, un mensaje de error (si lo hay) y la interpretación del gráfico.
        """
        if df is None or not (numeric_col in df.columns and category_col in df.columns):
            return None, "Selecciona una columna numérica y una categórica.", None
        if not pd.api.types.is_numeric_dtype(df[numeric_col]):
            return None, f"'{numeric_col}' debe ser numérica.", None
        if not (pd.api.types.is_categorical_dtype(df[category_col]) or pd.api.types.is_object_dtype(df[category_col])):
            return None, f"'{category_col}' debe ser categórica.", None
        
        temp_df = df[[numeric_col, category_col]].dropna()
        if temp_df.empty: return None, "No hay valores válidos para el boxplot agrupado.", None
        
        if temp_df[category_col].nunique() > 15: # Evitar demasiadas categorías en el eje X
            return None, f"La columna '{category_col}' tiene demasiadas categorías ({temp_df[category_col].nunique()}). Se recomiendan 15 o menos para un boxplot agrupado.", None

        fig, ax = plt.subplots(figsize=(min(12, max(6, temp_df[category_col].nunique()*0.8)), 6))
        sns.boxplot(x=temp_df[category_col], y=temp_df[numeric_col], ax=ax, palette="viridis")
        ax.set_title(f'Boxplot de {numeric_col} por {category_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel(category_col, fontsize=12)
        ax.set_ylabel(numeric_col, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()

        # --- Interpretación del Boxplot Agrupado ---
        interpretation = (
            f"Este **boxplot agrupado** compara la distribución de la variable numérica '{numeric_col}' "
            f"para cada categoría de la variable '{category_col}'. "
            f"Permite visualizar fácilmente las **diferencias en la mediana, dispersión y presencia de atípicos** "
            f"entre los distintos grupos, lo que es crucial para entender cómo una variable categórica influye en una numérica."
        )
        
        # Calcular medias y medianas por grupo para añadir a la interpretación
        means_by_group = temp_df.groupby(category_col)[numeric_col].mean().sort_values(ascending=False)
        medians_by_group = temp_df.groupby(category_col)[numeric_col].median().sort_values(ascending=False)

        interpretation += "\n\n**Estadísticas clave por grupo:**"
        for group in means_by_group.index:
            interpretation += f"\n- **{group}**: Media = {means_by_group[group]:.2f}, Mediana = {medians_by_group[group]:.2f}"
        
        interpretation += "\n\n**Observaciones:**"
        interpretation += "\n- Observa si las **cajas (rangos intercuartílicos)** se solapan significativamente. Un solapamiento menor sugiere que las distribuciones son estadísticamente diferentes."
        interpretation += "\n- Compara las **alturas de las cajas** para entender la variabilidad dentro de cada grupo."
        interpretation += "\n- Presta atención a la **presencia de atípicos** en ciertos grupos, lo que puede indicar casos especiales o errores de datos para ese grupo."
        interpretation += "\n- Si las medianas de los grupos son significativamente diferentes, esto sugiere que la variable categórica tiene un impacto real en la variable numérica."

        self.data_handler.add_analysis_event("Visualización", f"Boxplot Agrupado: '{numeric_col}' por '{category_col}'.", f"Medias por grupo: {means_by_group.to_dict()}.")
        return fig, None, interpretation

    def plot_correlation_heatmap(self, df: pd.DataFrame) -> tuple[plt.Figure | None, str | None, str | None]:
        """
        Genera un heatmap de la matriz de correlación para todas las columnas numéricas y proporciona su interpretación.
        Retorna la figura, un mensaje de error (si lo hay) y la interpretación del gráfico.
        """
        if df is None:
            return None, "Carga un conjunto de datos primero.", None
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.empty:
            return None, "No hay columnas numéricas para calcular la correlación.", None
        if len(numeric_df.columns) < 2:
            return None, "Se necesitan al menos dos columnas numéricas para calcular la correlación.", None
        
        corr_matrix = numeric_df.corr()
        if corr_matrix.empty:
            return None, "No se pudo calcular la matriz de correlación (quizás solo una columna numérica).", None

        fig, ax = plt.subplots(figsize=(len(corr_matrix.columns)*0.8 + 2, len(corr_matrix.columns)*0.8 + 2)) # Tamaño dinámico
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax, cbar_kws={'label': 'Coeficiente de Correlación'})
        ax.set_title('Matriz de Correlación', fontsize=14, fontweight='bold')
        fig.tight_layout()

        # --- Interpretación del Heatmap de Correlación ---
        interpretation = (
            f"El **heatmap de correlación** visualiza las relaciones lineales entre todas las variables numéricas del dataset. "
            f"Los valores en cada celda representan el **coeficiente de correlación de Pearson**, que varía de **-1 (correlación negativa perfecta)** "
            f"a **1 (correlación positiva perfecta)**. Un valor de **0 indica ausencia de correlación lineal**. "
            f"**Colores más cálidos (rojo)** indican una correlación positiva fuerte, "
            f"**colores más fríos (azul)** indican una correlación negativa fuerte, "
            f"y los **colores neutros (cerca de blanco/gris)** indican poca o ninguna correlación lineal."
        )
        
        strong_positive = []
        strong_negative = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)): # Evitar duplicados y diagonal
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                if corr_val > 0.7:
                    strong_positive.append(f"'{col1}' y '{col2}' (r={corr_val:.2f})")
                elif corr_val < -0.7:
                    strong_negative.append(f"'{col1}' y '{col2}' (r={corr_val:.2f})")
        
        if strong_positive:
            interpretation += "\n\n**Observaciones clave:**\n- Se identifican **fuertes correlaciones positivas** (las variables tienden a aumentar o disminuir juntas) entre: " + ", ".join(strong_positive) + "."
        if strong_negative:
            interpretation += "\n- Se identifican **fuertes correlaciones negativas** (cuando una variable aumenta, la otra tiende a disminuir) entre: " + ", ".join(strong_negative) + "."
        if not strong_positive and not strong_negative:
            interpretation += "\n\nNo se observan correlaciones lineales fuertes entre las variables numéricas en este dataset."

        interpretation += "\n\n**Importante:** Una alta correlación entre dos variables no implica necesariamente causalidad. Solo indica que hay una relación lineal entre ellas. Otros factores o variables latentes podrían estar influyendo en esta relación. Es un buen punto de partida para análisis más profundos."
        self.data_handler.add_analysis_event("Visualización", "Heatmap de Correlación.", f"Pares con correlación |r|>0.7: Positivas: {len(strong_positive)}, Negativas: {len(strong_negative)}.")
        return fig, None, interpretation

    def plot_barplot_categorical(self, df: pd.DataFrame, column: str) -> tuple[plt.Figure | None, str | None, str | None]:
        """
        Genera un gráfico de barras para una columna categórica y proporciona su interpretación.
        Retorna la figura, un mensaje de error (si lo hay) y la interpretación del gráfico.
        """
        if df is None or column not in df.columns or not (pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_bool_dtype(df[column])):
            return None, "Selecciona una columna categórica válida (o booleana) para el gráfico de barras.", None
        series = df[column].dropna()
        if series.empty: return None, f"La columna '{column}' no tiene valores categóricos válidos para graficar."

        value_counts = series.value_counts()
        if len(value_counts) > 20:
            return None, f"La columna '{column}' tiene demasiadas categorías ({len(value_counts)}). Considera agrupar o usar otro tipo de gráfico para una mejor visualización.", None

        fig, ax = plt.subplots(figsize=(min(12, max(6, len(value_counts)*0.8)), 6))
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax, palette="pastel")
        ax.set_title(f'Frecuencia de {column}', fontsize=14, fontweight='bold')
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel('Frecuencia', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()

        # --- Interpretación del Gráfico de Barras ---
        interpretation = (
            f"El **gráfico de barras para '{column}'** muestra la frecuencia (conteo de ocurrencias) de cada categoría única en la columna. "
            f"Es una herramienta fundamental para entender la **distribución de las categorías** y para identificar las categorías más y menos comunes, "
            f"así como para detectar desequilibrios en la variable."
        )
        
        most_common = value_counts.index[0]
        most_common_count = value_counts.values[0]
        total_obs = len(series)
        most_common_percent = (most_common_count / total_obs) * 100

        interpretation += f"\n\n- La categoría **'{most_common}'** es la más frecuente, con **{most_common_count} ocurrencias** ({most_common_percent:.1f}% del total)."
        if len(value_counts) > 1:
            least_common = value_counts.index[-1]
            least_common_count = value_counts.values[-1]
            least_common_percent = (least_common_count / total_obs) * 100
            interpretation += f"\n- La categoría **'{least_common}'** es la menos frecuente, con **{least_common_count} ocurrencias** ({least_common_percent:.1f}% del total)."
        
        interpretation += "\n\n**Observaciones:**"
        interpretation += "\n- Observa si la distribución es relativamente uniforme entre las categorías o si hay una o pocas categorías que dominan claramente el conjunto de datos (indicando un desequilibrio)."
        interpretation += "\n- Las categorías con muy pocas observaciones pueden ser problemáticas para algunos análisis y podrían requerir agrupamiento o un manejo especial."

        self.data_handler.add_analysis_event("Visualización", f"Gráfico de Barras: '{column}'.", f"Categoría más frecuente: '{most_common}' ({most_common_percent:.1f}%).")
        return fig, None, interpretation

    # --- Inferenciales Reales ---
    def run_one_sample_t_test(self, df: pd.DataFrame, column: str, pop_mean: float) -> str:
        """Realiza una prueba t de una muestra."""
        if df is None or column not in df.columns:
            return "Error: DataFrame no cargado o columna no válida."
        if not pd.api.types.is_numeric_dtype(df[column]):
            return "Error: La columna debe ser numérica para la prueba t."
        series = df[column].dropna()
        if series.empty:
            return "Error: La columna seleccionada no contiene datos numéricos válidos."
        
        try:
            t_stat, p_value = stats.ttest_1samp(series, pop_mean)
            output = f"--- Prueba t de una Muestra para '{column}' ---\n"
            output += f"Media muestral: {series.mean():.2f}\n"
            output += f"Hipótesis Nula (H0): La media poblacional es igual a {pop_mean}\n"
            output += f"Hipótesis Alterna (H1): La media poblacional es diferente de {pop_mean}\n"
            output += f"Estadístico t: {t_stat:.3f}\n"
            output += f"Valor p: {p_value:.3f}\n"
            
            conclusion = ""
            if p_value < 0.05: # Usando alfa=0.05 como umbral
                conclusion = "Se rechaza la hipótesis nula. Hay evidencia estadística significativa para concluir que la media poblacional de '{column}' es diferente de la media hipotetizada ({pop_mean})."
                output += "\nConclusión (alfa=0.05): " + conclusion
            else:
                conclusion = "No se rechaza la hipótesis nula. No hay evidencia estadística significativa para concluir que la media poblacional de '{column}' sea diferente de la media hipotetizada ({pop_mean})."
                output += "\nConclusión (alfa=0.05): " + conclusion
            
            self.data_handler.add_analysis_event(
                "Prueba Estadística",
                f"Prueba t de una muestra para '{column}' vs media={pop_mean}.",
                f"Estadístico t: {t_stat:.3f}, Valor p: {p_value:.3f}. Conclusión: {conclusion}"
            )
            return output
        except Exception as e:
            return f"Error al realizar la prueba t de una muestra: {e}"

    def run_independent_t_test(self, df: pd.DataFrame, numeric_col: str, group_col: str) -> str:
        """Realiza una prueba t para dos muestras independientes."""
        if df is None or not (numeric_col in df.columns and group_col in df.columns):
            return "Error: DataFrame no cargado o columnas no válidas."
        if not pd.api.types.is_numeric_dtype(df[numeric_col]):
            return "Error: La columna numérica debe ser numérica."
        if not (pd.api.types.is_categorical_dtype(df[group_col]) or pd.api.types.is_object_dtype(df[group_col])):
            return "Error: La columna de grupo debe ser categórica."
        
        groups = df[group_col].dropna().unique()
        if len(groups) != 2:
            return "Error: La columna de grupo debe tener exactamente dos categorías para una prueba t independiente."
        
        try:
            group1_data = df[df[group_col] == groups[0]][numeric_col].dropna()
            group2_data = df[df[group_col] == groups[1]][numeric_col].dropna()

            if group1_data.empty or group2_data.empty:
                return "Error: Uno o ambos grupos no tienen datos numéricos válidos."
            if len(group1_data) < 2 or len(group2_data) < 2:
                 return "Error: Se requieren al menos 2 observaciones por grupo para la prueba t."

            # Prueba de Levene para homogeneidad de varianzas
            levene_stat, levene_p = stats.levene(group1_data, group2_data)
            equal_var = levene_p > 0.05 # Si p > 0.05, asumimos varianzas iguales

            t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=equal_var)

            output = f"--- Prueba t para Muestras Independientes ---\n"
            output += f"Columna Numérica: '{numeric_col}'\n"
            output += f"Columna de Grupo: '{group_col}' (Grupos: {groups[0]} y {groups[1]})\n"
            output += f"Media de '{groups[0]}': {group1_data.mean():.2f} (n={len(group1_data)})\n"
            output += f"Media de '{groups[1]}': {group2_data.mean():.2f} (n={len(group2_data)})\n"
            output += f"\nPrueba de Levene para Homogeneidad de Varianzas:\n"
            output += f"  Estadístico F: {levene_stat:.3f}\n"
            output += f"  Valor p (Levene): {levene_p:.3f}\n"
            output += f"  Varianzas asumidas {'iguales' if equal_var else 'desiguales'}\n"
            output += f"\nResultados de la Prueba t:\n"
            output += f"  Estadístico t: {t_stat:.3f}\n"
            output += f"  Valor p: {p_value:.3f}\n"

            conclusion = ""
            if p_value < 0.05:
                conclusion = "Se rechaza la hipótesis nula. Hay una diferencia significativa entre las medias de los grupos de '{numeric_col}' según '{group_col}'."
                output += "\nConclusión (alfa=0.05): " + conclusion
            else:
                conclusion = "No se rechaza la hipótesis nula. No hay evidencia de una diferencia significativa entre las medias de los grupos de '{numeric_col}' según '{group_col}'."
                output += "\nConclusión (alfa=0.05): " + conclusion
            
            self.data_handler.add_analysis_event(
                "Prueba Estadística",
                f"Prueba t independiente para '{numeric_col}' por '{group_col}'.",
                f"Estadístico t: {t_stat:.3f}, Valor p: {p_value:.3f}, Grupos: {groups[0]} vs {groups[1]}. Conclusión: {conclusion}"
            )
            return output
        except Exception as e:
            return f"Error al realizar la prueba t independiente: {e}"

    def run_simple_linear_regression(self, df: pd.DataFrame, dependent_var: str, independent_var: str) -> str:
        """Realiza una regresión lineal simple."""
        if df is None or not (dependent_var in df.columns and independent_var in df.columns):
            return "Error: DataFrame no cargado o columnas no válidas."
        if not (pd.api.types.is_numeric_dtype(df[dependent_var]) and pd.api.types.is_numeric_dtype(df[independent_var])):
            return "Error: Ambas columnas deben ser numéricas para la regresión lineal."
        
        temp_df = df[[dependent_var, independent_var]].dropna()
        if temp_df.empty:
            return "Error: No hay datos válidos en las columnas seleccionadas para la regresión."
        if len(temp_df) < 2: # Necesita al menos 2 puntos para una línea
            return "Error: Se necesitan al menos dos pares de observaciones válidas para la regresión."

        try:
            X = sm.add_constant(temp_df[independent_var]) # Añadir constante para intercepto
            y = temp_df[dependent_var]
            model = sm.OLS(y, X).fit()
            
            output = f"--- Resultados de la Regresión Lineal Simple ---\n"
            output += f"Variable Dependiente (Y): '{dependent_var}'\n"
            output += f"Variable Independiente (X): '{independent_var}'\n\n"
            output += model.summary().as_text()

            r_squared = model.rsquared
            p_value_indep = model.pvalues[independent_var]
            
            conclusion = f"El modelo de regresión lineal simple busca predecir '{dependent_var}' usando '{independent_var}'.\n"
            conclusion += f"- El **R-cuadrado ({r_squared:.3f})** indica que el {r_squared*100:.1f}% de la varianza en '{dependent_var}' es explicada por '{independent_var}'.\n"
            
            if p_value_indep < 0.05:
                conclusion += f"- La variable independiente '{independent_var}' es **estadísticamente significativa** (p={p_value_indep:.3f}) para predecir '{dependent_var}'.\n"
                conclusion += f"- El coeficiente para '{independent_var}' es **{model.params[independent_var]:.3f}**. Esto significa que por cada unidad que aumenta '{independent_var}', '{dependent_var}' se espera que cambie en {model.params[independent_var]:.3f} unidades (manteniendo todo lo demás constante)."
            else:
                conclusion += f"- La variable independiente '{independent_var}' **no es estadísticamente significativa** (p={p_value_indep:.3f}) para predecir '{dependent_var}' en este modelo."

            self.data_handler.add_analysis_event(
                "Prueba Estadística",
                f"Regresión Lineal Simple: '{dependent_var}' ~ '{independent_var}'.",
                f"R-cuadrado: {r_squared:.3f}, p-valor de '{independent_var}': {p_value_indep:.3f}. Conclusión: {conclusion.split('.')[0]}."
            )
            return output + "\n\n" + conclusion
        except Exception as e:
            return f"Error al realizar la regresión lineal simple: {e}"

    def run_multiple_linear_regression(self, df: pd.DataFrame, dependent_var: str, independent_vars: list[str]) -> str:
        """Realiza una regresión lineal múltiple."""
        if df is None or not dependent_var in df.columns:
            return "Error: DataFrame no cargado o variable dependiente no válida."
        if not independent_vars:
            return "Error: Seleccione al menos una variable independiente."
        
        for col in [dependent_var] + independent_vars:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return f"Error: La columna '{col}' debe ser numérica para la regresión."

        cols_to_use = [dependent_var] + independent_vars
        temp_df = df[cols_to_use].dropna()

        if temp_df.empty:
            return "Error: No hay datos válidos en las columnas seleccionadas para la regresión múltiple."
        if len(temp_df) < len(independent_vars) + 1:
            return "Error: Se necesitan más observaciones que el número de predictores (incluyendo la constante) para una regresión múltiple válida."

        try:
            X = sm.add_constant(temp_df[independent_vars]) # Añadir constante para intercepto
            y = temp_df[dependent_var]
            model = sm.OLS(y, X).fit()
            
            output = f"--- Resultados de la Regresión Lineal Múltiple ---\n"
            output += f"Variable Dependiente (Y): '{dependent_var}'\n"
            output += f"Variables Independientes (X): {', '.join(independent_vars)}\n\n"
            output += model.summary().as_text()

            r_squared = model.rsquared
            adj_r_squared = model.rsquared_adj
            f_pvalue = model.f_pvalue
            
            conclusion = f"El modelo de regresión lineal múltiple busca predecir '{dependent_var}' utilizando múltiples variables independientes.\n"
            conclusion += f"- El **R-cuadrado ({r_squared:.3f})** indica que el {r_squared*100:.1f}% de la varianza en '{dependent_var}' es explicada por todas las variables independientes en conjunto.\n"
            conclusion += f"- El **R-cuadrado ajustado ({adj_r_squared:.3f})** es una medida más conservadora, útil al comparar modelos con diferente número de predictores.\n"
            
            if f_pvalue < 0.05:
                conclusion += f"- El modelo es **estadísticamente significativo** (p-valor del F-test={f_pvalue:.3f}), lo que sugiere que al menos una de las variables independientes predice significativamente la variable dependiente.\n"
            else:
                conclusion += f"- El modelo **no es estadísticamente significativo** (p-valor del F-test={f_pvalue:.3f}), lo que implica que el conjunto de variables independientes no predice significativamente la variable dependiente."
            
            conclusion += "\n**Significación de predictores individuales (observando p-valores de los coeficientes):**\n"
            for var, p_val in model.pvalues.drop('const').items():
                coeff = model.params[var]
                if p_val < 0.05:
                    conclusion += f"- '{var}': Coeficiente={coeff:.3f}, p-valor={p_val:.3f}. Es **significativo**. Un aumento de una unidad en '{var}' se asocia con un cambio de {coeff:.3f} en '{dependent_var}', manteniendo las otras variables constantes.\n"
                else:
                    conclusion += f"- '{var}': Coeficiente={coeff:.3f}, p-valor={p_val:.3f}. **No es significativo**."
            
            self.data_handler.add_analysis_event(
                "Prueba Estadística",
                f"Regresión Lineal Múltiple: '{dependent_var}' ~ {', '.join(independent_vars)}.",
                f"R-cuadrado: {r_squared:.3f}, p-valor F: {f_pvalue:.3f}. Conclusión: {conclusion.split('.')[0]}."
            )
            return output + "\n\n" + conclusion
        except Exception as e:
            return f"Error al realizar la regresión lineal múltiple: {e}"

    def run_anova(self, df: pd.DataFrame, numeric_col: str, group_col: str) -> str:
        """Realiza un Análisis de Varianza (ANOVA) de un factor."""
        if df is None or not (numeric_col in df.columns and group_col in df.columns):
            return "Error: DataFrame no cargado o columnas no válidas."
        if not pd.api.types.is_numeric_dtype(df[numeric_col]):
            return "Error: La columna numérica debe ser numérica para ANOVA."
        if not (pd.api.types.is_categorical_dtype(df[group_col]) or pd.api.types.is_object_dtype(df[group_col])):
            return "Error: La columna de grupo debe ser categórica para ANOVA."
        
        groups = df[group_col].dropna().unique()
        if len(groups) < 2:
            return "Error: La columna de grupo debe tener al menos dos categorías para ANOVA."
        
        data_per_group = [df[df[group_col] == g][numeric_col].dropna() for g in groups]
        
        # Filtrar grupos vacíos o con solo una observación
        valid_groups_data = [data for data in data_per_group if len(data) >= 2]
        valid_groups_names = [groups[i] for i, data in enumerate(data_per_group) if len(data) >= 2]

        if len(valid_groups_data) < 2:
            return "Error: Necesita al menos dos grupos con al menos 2 observaciones válidas cada uno para ANOVA."

        try:
            f_statistic, p_value = stats.f_oneway(*valid_groups_data)
            
            output = f"--- Análisis de Varianza (ANOVA) para '{numeric_col}' por '{group_col}' ---\n"
            output += f"Columna Numérica: '{numeric_col}'\n"
            output += f"Columna de Grupo: '{group_col}' (Grupos analizados: {', '.join(map(str, valid_groups_names))})\n\n"
            output += f"Estadístico F: {f_statistic:.3f}\n"
            output += f"Valor p: {p_value:.3f}\n"

            conclusion = ""
            if p_value < 0.05:
                conclusion = "Se rechaza la hipótesis nula. Hay una diferencia significativa entre las medias de al menos dos de los grupos de '{numeric_col}' definidos por '{group_col}'. Esto sugiere que el factor categórico tiene un efecto significativo en la variable numérica."
                output += "\nConclusión (alfa=0.05): " + conclusion
            else:
                conclusion = "No se rechaza la hipótesis nula. No hay evidencia de una diferencia significativa entre las medias de los grupos de '{numeric_col}' definidos por '{group_col}'. Esto sugiere que el factor categórico no tiene un efecto significativo en la variable numérica, o que el tamaño de la muestra no es suficiente para detectarlo."
                output += "\nConclusión (alfa=0.05): " + conclusion
            
            # Registrar para meta-análisis
            mean_values = {str(g): data.mean() for g, data in zip(valid_groups_names, valid_groups_data)}
            self.data_handler.add_analysis_event(
                "Prueba Estadística",
                f"ANOVA: '{numeric_col}' por '{group_col}'.",
                f"Estadístico F: {f_statistic:.3f}, Valor p: {p_value:.3f}. Medias: {mean_values}. Conclusión: {conclusion}"
            )
            return output
        except Exception as e:
            return f"Error al realizar ANOVA: {e}"

    def run_chi_squared_test(self, df: pd.DataFrame, col1: str, col2: str) -> str:
        """Realiza una prueba Chi-cuadrado de independencia."""
        if df is None or not (col1 in df.columns and col2 in df.columns):
            return "Error: DataFrame no cargado o columnas no válidas."
        if not (pd.api.types.is_categorical_dtype(df[col1]) or pd.api.types.is_object_dtype(df[col1]) or pd.api.types.is_bool_dtype(df[col1])):
            return f"Error: La columna '{col1}' debe ser categórica (o booleana)."
        if not (pd.api.types.is_categorical_dtype(df[col2]) or pd.api.types.is_object_dtype(df[col2]) or pd.api.types.is_bool_dtype(df[col2])):
            return f"Error: La columna '{col2}' debe ser categórica (o booleana)."
        
        temp_df = df[[col1, col2]].dropna()
        if temp_df.empty:
            return "Error: No hay datos válidos en las columnas seleccionadas para la prueba Chi-cuadrado."

        try:
            contingency_table = pd.crosstab(temp_df[col1], temp_df[col2])
            if contingency_table.min().min() == 0:
                return "Advertencia: Una o más celdas en la tabla de contingencia tienen una frecuencia observada de cero. Esto podría afectar la validez de la prueba Chi-cuadrado si las frecuencias esperadas también son muy bajas (idealmente, todas las esperadas > 5)."

            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

            output = f"--- Prueba Chi-cuadrado de Independencia ---\n"
            output += f"Columnas analizadas: '{col1}' y '{col2}'\n\n"
            output += f"Tabla de Contingencia (Observada):\n{contingency_table.to_string()}\n\n"
            output += f"Estadístico Chi-cuadrado: {chi2:.3f}\n"
            output += f"Valor p: {p_value:.3f}\n"
            output += f"Grados de Libertad (dof): {dof}\n"
            output += f"Frecuencias Esperadas:\n{pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns).round(2).to_string()}\n"

            conclusion = ""
            if p_value < 0.05:
                conclusion = "Se rechaza la hipótesis nula. Hay una asociación significativa entre las dos variables categóricas ('{col1}' y '{col2}'). Esto significa que la distribución de una variable no es independiente de la otra."
                output += "\nConclusión (alfa=0.05): " + conclusion
            else:
                conclusion = "No se rechaza la hipótesis nula. No hay evidencia de una asociación significativa entre las dos variables categóricas ('{col1}' y '{col2}'). Esto sugiere que las variables son independientes."
                output += "\nConclusión (alfa=0.05): " + conclusion
            
            self.data_handler.add_analysis_event(
                "Prueba Estadística",
                f"Chi-cuadrado: '{col1}' vs '{col2}'.",
                f"Chi2: {chi2:.3f}, p-valor: {p_value:.3f}. Conclusión: {conclusion}"
            )
            return output
        except Exception as e:
            return f"Error al realizar la prueba Chi-cuadrado: {e}"


# --- main_app.py (Clase Principal de la Aplicación Tkinter) ---
class ProStatApp(tk.Tk):
    """
    Clase principal de la aplicación de Análisis Estadístico Profesional.
    Proporciona una GUI para carga de datos, estadísticas descriptivas, gráficos básicos,
    y marcadores de posición para análisis avanzados.
    """
    def __init__(self):
        super().__init__()
        self.title("Análisis Estadístico Profesional")
        self.geometry("1400x900")
        self.minsize(1000, 700)
        
        self.data_handler = DataHandler()
        self.stats_analyzer = StatsAnalyzer(self.data_handler) # Pasar data_handler al StatsAnalyzer

        self.style = ttk.Style()
        apply_custom_styles(self.style)

        self._create_icon() # Crear y aplicar el ícono
        self._setup_ui()

        # Almacenar la figura actual y su interpretación para exportación
        self.current_plot_fig = None
        self.current_plot_interpretation = None
        self.current_plot_title = None

    def _create_icon(self):
        """Crea un pequeño icono circular con las letras 'PS'."""
        try:
            # Crear una imagen pequeña en memoria
            img_size = 64
            img = Image.new('RGBA', (img_size, img_size), (0, 0, 0, 0)) # Transparente
            draw = ImageDraw.Draw(img)

            # Dibujar un círculo
            draw.ellipse((0, 0, img_size, img_size), fill="#3498DB", outline="#2C3E50", width=2) # Azul Cielo

            # Añadir texto "PS"
            try:
                font_path = ImageFont.truetype("arial.ttf", int(img_size * 0.5)) # Intentar cargar Arial
            except IOError:
                font_path = ImageFont.load_default() # Fallback a fuente predeterminada
            
            text = "PS"
            text_bbox = draw.textbbox((0, 0), text, font=font_path)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_x = (img_size - text_width) / 2
            text_y = (img_size - text_height) / 2 - (img_size * 0.05) # Pequeño ajuste vertical

            draw.text((text_x, text_y), text, font=font_path, fill="white")
            
            # Guardar en un buffer para tkinter
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            self.iconphoto(True, tk.PhotoImage(data=buffer.getvalue()))
        except Exception as e:
            print(f"No se pudo crear el ícono de la aplicación: {e}")

    def _setup_ui(self):
        """Configura la estructura principal de la interfaz de usuario."""
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        # Pestañas
        self.tab_data_load = ttk.Frame(self.notebook)
        self.tab_descriptive = ttk.Frame(self.notebook)
        self.tab_visualize = ttk.Frame(self.notebook)
        self.tab_data_prep = ttk.Frame(self.notebook)
        self.tab_inferential = ttk.Frame(self.notebook)
        self.tab_meta_analysis = ttk.Frame(self.notebook) # Nueva pestaña para meta-análisis

        self.notebook.add(self.tab_data_load, text="Cargar Datos")
        self.notebook.add(self.tab_data_prep, text="Preparar Datos")
        self.notebook.add(self.tab_descriptive, text="Est. Descriptivas")
        self.notebook.add(self.tab_visualize, text="Visualización")
        self.notebook.add(self.tab_inferential, text="Est. Inferencial")
        self.notebook.add(self.tab_meta_analysis, text="Meta-Análisis") # Añadir pestaña

        self._create_data_load_tab()
        self._create_data_prep_tab()
        self._create_descriptive_tab()
        self._create_visualize_tab()
        self._create_inferential_tab()
        self._create_meta_analysis_tab() # Crear widgets para la nueva pestaña

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)

    def _on_tab_change(self, event):
        """Actualiza la interfaz cuando cambia la pestaña activa."""
        selected_tab = self.notebook.tab(self.notebook.select(), "text")
        if selected_tab == "Est. Descriptivas" and self.data_handler.get_dataframe() is not None:
            self._update_column_listbox(self.descriptive_column_listbox)
        elif selected_tab == "Visualización" and self.data_handler.get_dataframe() is not None:
            self._update_plot_controls()
        elif selected_tab == "Preparar Datos" and self.data_handler.get_dataframe() is not None:
            self._update_data_prep_controls()
        elif selected_tab == "Est. Inferencial" and self.data_handler.get_dataframe() is not None:
            self._update_inferential_controls()
        elif selected_tab == "Meta-Análisis":
            self._update_meta_analysis_content()


    # --- Pestaña de Carga de Datos ---
    def _create_data_load_tab(self):
        frame = ttk.Frame(self.tab_data_load, padding="20")
        frame.pack(expand=True, fill="both")

        ttk.Label(frame, text="Cargar Datos del Archivo", style="Header.TLabel").pack(pady=20)

        load_button = ttk.Button(frame, text="Seleccionar Archivo (CSV/XLSX)", command=self._load_file)
        load_button.pack(pady=10)

        self.file_path_label = ttk.Label(frame, text="No hay archivo cargado.", style="Info.TLabel")
        self.file_path_label.pack(pady=5)

        ttk.Label(frame, text="Previsualización de Datos", style="SubHeader.TLabel").pack(pady=15)

        # Frame para la tabla Treeview
        table_frame = ttk.Frame(frame)
        table_frame.pack(expand=True, fill="both", padx=10, pady=5)

        self.data_tree = ttk.Treeview(table_frame, show="headings")
        self.data_tree.pack(side="left", expand=True, fill="both")

        # Scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.data_tree.yview)
        vsb.pack(side="right", fill="y")
        self.data_tree.configure(yscrollcommand=vsb.set)

        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.data_tree.xview)
        hsb.pack(side="bottom", fill="x")
        self.data_tree.configure(xscrollcommand=hsb.set)

        # Botones de Sesión
        session_frame = ttk.Frame(frame)
        session_frame.pack(pady=20)
        
        ttk.Button(session_frame, text="Guardar Sesión", command=self._save_session).pack(side="left", padx=5)
        ttk.Button(session_frame, text="Cargar Sesión", command=self._load_session).pack(side="left", padx=5)

    def _load_file(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Archivos de Datos", "*.csv *.xlsx *.xls"), ("Todos los archivos", "*.*")]
        )
        if filepath:
            try:
                self.data_handler.load_data(filepath)
                self.file_path_label.config(text=f"Archivo cargado: {os.path.basename(filepath)}")
                self._update_data_preview(self.data_handler.get_dataframe())
                messagebox.showinfo("Éxito", "Datos cargados correctamente.")
                self.notebook.select(self.tab_descriptive) # Mover a tab de descriptivos
            except Exception as e:
                messagebox.showerror("Error al cargar", f"No se pudo cargar el archivo: {e}")
                self.data_handler._reset_data()
                self.file_path_label.config(text="No hay archivo cargado.")
                self._update_data_preview(None)

    def _update_data_preview(self, df: pd.DataFrame | None):
        """Actualiza el Treeview con los datos del DataFrame."""
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        self.data_tree["columns"] = ()

        if df is not None:
            self.data_tree["columns"] = df.columns.tolist()
            for col in df.columns:
                self.data_tree.heading(col, text=col)
                self.data_tree.column(col, width=100, anchor="w")

            # Mostrar solo las primeras 100 filas para previsualización
            for index, row in df.head(100).iterrows():
                self.data_tree.insert("", "end", values=row.tolist())
        else:
            self.data_tree["columns"] = ["Columna", "Tipo de Dato"]
            self.data_tree.heading("Columna", text="Columna")
            self.data_tree.heading("Tipo de Dato", text="Tipo de Dato")
            self.data_tree.insert("", "end", values=["No hay datos cargados", ""])
    
    def _save_session(self):
        if self.data_handler.get_dataframe() is None:
            messagebox.showwarning("Advertencia", "No hay datos cargados para guardar la sesión.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Archivos de Sesión", "*.pkl")]
        )
        if filepath:
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump({
                        'df': self.data_handler.df,
                        'original_df': self.data_handler.original_df,
                        'filepath': self.data_handler.filepath,
                        'analysis_history': self.data_handler.analysis_history
                    }, f)
                messagebox.showinfo("Éxito", "Sesión guardada correctamente.")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar la sesión: {e}")

    def _load_session(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Archivos de Sesión", "*.pkl")]
        )
        if filepath:
            try:
                with open(filepath, 'rb') as f:
                    session_data = pickle.load(f)
                    self.data_handler.df = session_data.get('df')
                    self.data_handler.original_df = session_data.get('original_df')
                    self.data_handler.filepath = session_data.get('filepath')
                    self.data_handler.analysis_history = session_data.get('analysis_history', []) # Cargar historial
                
                self._update_data_preview(self.data_handler.get_dataframe())
                if self.data_handler.filepath:
                    self.file_path_label.config(text=f"Sesión cargada: {os.path.basename(self.data_handler.filepath)}")
                else:
                    self.file_path_label.config(text="Sesión cargada (archivo original no especificado).")
                messagebox.showinfo("Éxito", "Sesión cargada correctamente.")
                self.notebook.select(self.tab_descriptive) # Mover a tab de descriptivos
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar la sesión: {e}")
                self.data_handler._reset_data()
                self.file_path_label.config(text="No hay archivo cargado.")
                self._update_data_preview(None)

    # --- Pestaña de Preparación de Datos ---
    def _create_data_prep_tab(self):
        frame = ttk.Frame(self.tab_data_prep, padding="20")
        frame.pack(expand=True, fill="both")

        ttk.Label(frame, text="Preparación de Datos", style="Header.TLabel").pack(pady=10)

        # Información general del DataFrame
        info_frame = ttk.LabelFrame(frame, text="Información del Dataset", padding="10")
        info_frame.pack(fill="x", pady=10)
        self.df_info_text = scrolledtext.ScrolledText(info_frame, wrap=tk.WORD, height=8, font=("Segoe UI", 9), relief="flat")
        self.df_info_text.pack(fill="both", expand=True)
        self.df_info_text.config(state=tk.DISABLED)

        # Manejo de valores faltantes
        missing_frame = ttk.LabelFrame(frame, text="Manejo de Valores Faltantes", padding="10")
        missing_frame.pack(fill="x", pady=10)

        ttk.Label(missing_frame, text="Columna(s):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.missing_col_listbox = tk.Listbox(missing_frame, selectmode=tk.MULTIPLE, height=5, exportselection=0, font=("Segoe UI", 9))
        self.missing_col_listbox.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        missing_col_scrollbar = ttk.Scrollbar(missing_frame, orient="vertical", command=self.missing_col_listbox.yview)
        missing_col_scrollbar.grid(row=1, column=2, sticky="ns")
        self.missing_col_listbox.config(yscrollcommand=missing_col_scrollbar.set)
        
        ttk.Label(missing_frame, text="Estrategia:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.missing_strategy_var = tk.StringVar(value="drop_rows")
        strategy_options = ["drop_rows", "drop_columns", "impute_mean", "impute_median", "impute_mode", "impute_value"]
        strategy_menu = ttk.OptionMenu(missing_frame, self.missing_strategy_var, self.missing_strategy_var.get(), *strategy_options, command=self._toggle_impute_value_entry)
        strategy_menu.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.impute_value_label = ttk.Label(missing_frame, text="Valor a Imputar:", state=tk.DISABLED)
        self.impute_value_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.impute_value_entry = ttk.Entry(missing_frame, state=tk.DISABLED)
        self.impute_value_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        ttk.Button(missing_frame, text="Aplicar Manejo de Nulos", command=self._apply_missing_value_handling).grid(row=4, column=0, columnspan=3, pady=10)

        missing_frame.grid_columnconfigure(1, weight=1)

    def _toggle_impute_value_entry(self, *args):
        if self.missing_strategy_var.get() == "impute_value":
            self.impute_value_label.config(state=tk.NORMAL)
            self.impute_value_entry.config(state=tk.NORMAL)
        else:
            self.impute_value_label.config(state=tk.DISABLED)
            self.impute_value_entry.config(state=tk.DISABLED)

    def _update_data_prep_controls(self):
        df = self.data_handler.get_dataframe()
        if df is None:
            self.df_info_text.config(state=tk.NORMAL)
            self.df_info_text.delete(1.0, tk.END)
            self.df_info_text.insert(tk.END, "No hay datos cargados.")
            self.df_info_text.config(state=tk.DISABLED)
            self.missing_col_listbox.delete(0, tk.END)
            return

        # Actualizar info del DF
        output_buffer = io.StringIO()
        with contextlib.redirect_stdout(output_buffer):
            df.info()
            print("\n")
            print("Tipos de Datos:\n", df.dtypes.to_string())
            missing_info = self.data_handler.get_missing_values_info()
            if missing_info is not None and not missing_info.empty:
                print("\nValores Faltantes:\n", missing_info.to_string())
            else:
                print("\nNo hay valores faltantes.")
        self.df_info_text.config(state=tk.NORMAL)
        self.df_info_text.delete(1.0, tk.END)
        self.df_info_text.insert(tk.END, output_buffer.getvalue())
        self.df_info_text.config(state=tk.DISABLED)

        # Actualizar lista de columnas para manejo de nulos
        self.missing_col_listbox.delete(0, tk.END)
        for col in self.data_handler.get_column_names():
            self.missing_col_listbox.insert(tk.END, col)
            if df[col].isnull().any(): # Seleccionar columnas con nulos por defecto
                self.missing_col_listbox.selection_set(tk.END)


    def _apply_missing_value_handling(self):
        if self.data_handler.get_dataframe() is None:
            messagebox.showwarning("Advertencia", "Carga un archivo de datos primero.")
            return
        
        selected_indices = self.missing_col_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Advertencia", "Selecciona al menos una columna para aplicar el manejo de nulos.")
            return
        
        selected_columns = [self.missing_col_listbox.get(i) for i in selected_indices]
        strategy = self.missing_strategy_var.get()
        value = self.impute_value_entry.get() if strategy == "impute_value" else None

        try:
            # Algunas estrategias (drop_rows, drop_columns) aplican a todas las seleccionadas de golpe
            # Para imputación, la DataHandler itera si se le pasan múltiples columnas.
            result_msg = self.data_handler.handle_missing_values(strategy, selected_columns, value)
            messagebox.showinfo("Operación Exitosa", result_msg)
            self._update_data_preview(self.data_handler.get_dataframe()) # Actualizar vista previa
            self._update_data_prep_controls() # Refrescar información de nulos
            self._update_plot_controls() # Para que se actualicen las listas de columnas en visualización
            self._update_inferential_controls() # Para que se actualicen las listas de columnas en inferencial
        except Exception as e:
            messagebox.showerror("Error al aplicar", f"Hubo un error: {e}")

    # --- Pestaña de Estadísticas Descriptivas ---
    def _create_descriptive_tab(self):
        frame = ttk.Frame(self.tab_descriptive, padding="20")
        frame.pack(expand=True, fill="both")

        ttk.Label(frame, text="Estadísticas Descriptivas", style="Header.TLabel").pack(pady=10)

        control_frame = ttk.Frame(frame)
        control_frame.pack(pady=10, fill="x")

        ttk.Label(control_frame, text="Selecciona Columna:").pack(side="left", padx=5)
        
        self.descriptive_column_var = tk.StringVar()
        self.descriptive_column_menu = ttk.OptionMenu(control_frame, self.descriptive_column_var, "")
        self.descriptive_column_menu.pack(side="left", padx=5, expand=True, fill="x")
        
        analyze_button = ttk.Button(control_frame, text="Calcular", command=self._run_descriptive_stats)
        analyze_button.pack(side="left", padx=5)

        self.descriptive_output_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=20, font=("Courier New", 10), relief="groove")
        self.descriptive_output_text.pack(expand=True, fill="both", pady=10)
        self.descriptive_output_text.config(state=tk.DISABLED)

    def _update_column_listbox(self, listbox_widget: tk.Listbox | None = None, dropdown_menu: ttk.OptionMenu | None = None, string_var: tk.StringVar | None = None, column_type: str = "all"):
        """
        Actualiza los widgets de selección de columna con las columnas del DataFrame.
        column_type: "all", "numeric", "categorical"
        """
        df = self.data_handler.get_dataframe()
        if df is None:
            cols = []
        else:
            if column_type == "numeric":
                cols = self.data_handler.get_numeric_columns(df)
            elif column_type == "categorical":
                cols = self.data_handler.get_categorical_columns(df)
            else:
                cols = self.data_handler.get_column_names(df)

        if listbox_widget:
            listbox_widget.delete(0, tk.END)
            for col in cols:
                listbox_widget.insert(tk.END, col)
        
        if dropdown_menu and string_var:
            menu = dropdown_menu["menu"]
            menu.delete(0, "end")
            if not cols:
                string_var.set("")
                menu.add_command(label="No hay columnas", command=lambda: None)
            else:
                string_var.set(cols[0]) # Set default to first column
                for col in cols:
                    menu.add_command(label=col, command=tk._setit(string_var, col))

    def _run_descriptive_stats(self):
        if self.data_handler.get_dataframe() is None:
            messagebox.showwarning("Advertencia", "Carga un archivo de datos primero.")
            return
        
        selected_column = self.descriptive_column_var.get()
        if not selected_column:
            messagebox.showwarning("Advertencia", "Selecciona una columna para el análisis descriptivo.")
            return

        df = self.data_handler.get_dataframe()
        output = self.stats_analyzer.get_descriptive_stats(df, selected_column)
        
        self.descriptive_output_text.config(state=tk.NORMAL)
        self.descriptive_output_text.delete(1.0, tk.END)
        self.descriptive_output_text.insert(tk.END, output)
        self.descriptive_output_text.config(state=tk.DISABLED)

    # --- Pestaña de Visualización ---
    def _create_visualize_tab(self):
        main_frame = ttk.Frame(self.tab_visualize, padding="15")
        main_frame.pack(expand=True, fill="both")

        # Controles de gráfico (izquierda)
        controls_frame = ttk.LabelFrame(main_frame, text="Controles de Gráfico", padding="10")
        controls_frame.pack(side="left", fill="y", padx=10, pady=10)

        # Tipo de gráfico
        ttk.Label(controls_frame, text="Tipo de Gráfico:").pack(pady=5, anchor="w")
        self.plot_type_var = tk.StringVar(value="Histograma")
        plot_types = ["Histograma", "Boxplot", "Diagrama de Dispersión", "Boxplot Agrupado", "Heatmap de Correlación", "Gráfico de Barras (Categórica)"]
        ttk.OptionMenu(controls_frame, self.plot_type_var, self.plot_type_var.get(), *plot_types, command=self._on_plot_type_change).pack(pady=5, fill="x")

        # Controles específicos para cada tipo de gráfico
        self.plot_specific_controls_frame = ttk.Frame(controls_frame)
        self.plot_specific_controls_frame.pack(pady=10, fill="both", expand=True)
        self._on_plot_type_change(self.plot_type_var.get()) # Inicializar controles

        ttk.Button(controls_frame, text="Generar Gráfico", command=self.generate_plot).pack(pady=10, fill="x")
        ttk.Button(controls_frame, text="Exportar Gráfico y Análisis a PDF", command=self._export_current_plot_to_pdf).pack(pady=5, fill="x")

        # Sección de Visualización (derecha)
        plot_display_frame = ttk.Frame(main_frame, style="PlotFrame.TFrame", padding="15")
        plot_display_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Contenedor para el gráfico
        self.plot_canvas_container = ttk.Frame(plot_display_frame)
        self.plot_canvas_container.pack(side="top", fill="both", expand=True, pady=(0, 10))

        # Etiqueta para el título del gráfico
        self.plot_title_label = ttk.Label(plot_display_frame, text="Gráfico", style="SubHeader.TLabel")
        self.plot_title_label.pack(side="top", fill="x", pady=(5, 5))

        # Área de texto para la interpretación
        self.plot_interpretation_text = scrolledtext.ScrolledText(plot_display_frame, wrap=tk.WORD, height=8, font=("Segoe UI", 9), relief="solid", borderwidth=1, bd=1)
        self.plot_interpretation_text.pack(side="bottom", fill="x", pady=(5, 0))
        self.plot_interpretation_text.config(state=tk.DISABLED)

    def _on_plot_type_change(self, selected_type):
        """Actualiza los controles específicos del gráfico según el tipo seleccionado."""
        # Limpiar controles anteriores
        for widget in self.plot_specific_controls_frame.winfo_children():
            widget.destroy()

        self.plot_column_var = tk.StringVar() # Para Histograma, Boxplot, Barplot (Categórica)
        self.plot_x_column_var = tk.StringVar() # Para Scatterplot
        self.plot_y_column_var = tk.StringVar() # Para Scatterplot
        self.plot_numeric_col_var = tk.StringVar() # Para Boxplot Agrupado
        self.plot_category_col_var = tk.StringVar() # Para Boxplot Agrupado

        df = self.data_handler.get_dataframe()
        all_cols = self.data_handler.get_column_names(df)
        numeric_cols = self.data_handler.get_numeric_columns(df)
        categorical_cols = self.data_handler.get_categorical_columns(df)

        if selected_type in ["Histograma", "Boxplot"]:
            ttk.Label(self.plot_specific_controls_frame, text="Columna Numérica:").pack(pady=5, anchor="w")
            self.plot_column_menu = ttk.OptionMenu(self.plot_specific_controls_frame, self.plot_column_var, "", *numeric_cols)
            self.plot_column_menu.pack(pady=5, fill="x")
            self._update_column_listbox(dropdown_menu=self.plot_column_menu, string_var=self.plot_column_var, column_type="numeric")
        
        elif selected_type == "Diagrama de Dispersión":
            ttk.Label(self.plot_specific_controls_frame, text="Columna X (Numérica):").pack(pady=5, anchor="w")
            self.plot_x_column_menu = ttk.OptionMenu(self.plot_specific_controls_frame, self.plot_x_column_var, "", *numeric_cols)
            self.plot_x_column_menu.pack(pady=5, fill="x")
            self._update_column_listbox(dropdown_menu=self.plot_x_column_menu, string_var=self.plot_x_column_var, column_type="numeric")

            ttk.Label(self.plot_specific_controls_frame, text="Columna Y (Numérica):").pack(pady=5, anchor="w")
            self.plot_y_column_menu = ttk.OptionMenu(self.plot_specific_controls_frame, self.plot_y_column_var, "", *numeric_cols)
            self.plot_y_column_menu.pack(pady=5, fill="x")
            self._update_column_listbox(dropdown_menu=self.plot_y_column_menu, string_var=self.plot_y_column_var, column_type="numeric")

        elif selected_type == "Boxplot Agrupado":
            ttk.Label(self.plot_specific_controls_frame, text="Columna Numérica:").pack(pady=5, anchor="w")
            self.plot_numeric_col_menu = ttk.OptionMenu(self.plot_specific_controls_frame, self.plot_numeric_col_var, "", *numeric_cols)
            self.plot_numeric_col_menu.pack(pady=5, fill="x")
            self._update_column_listbox(dropdown_menu=self.plot_numeric_col_menu, string_var=self.plot_numeric_col_var, column_type="numeric")

            ttk.Label(self.plot_specific_controls_frame, text="Columna Categórica (Grupo):").pack(pady=5, anchor="w")
            self.plot_category_col_menu = ttk.OptionMenu(self.plot_specific_controls_frame, self.plot_category_col_var, "", *categorical_cols)
            self.plot_category_col_menu.pack(pady=5, fill="x")
            self._update_column_listbox(dropdown_menu=self.plot_category_col_menu, string_var=self.plot_category_col_var, column_type="categorical")

        elif selected_type == "Gráfico de Barras (Categórica)":
            ttk.Label(self.plot_specific_controls_frame, text="Columna Categórica:").pack(pady=5, anchor="w")
            self.plot_column_menu = ttk.OptionMenu(self.plot_specific_controls_frame, self.plot_column_var, "", *categorical_cols)
            self.plot_column_menu.pack(pady=5, fill="x")
            self._update_column_listbox(dropdown_menu=self.plot_column_menu, string_var=self.plot_column_var, column_type="categorical")

        elif selected_type == "Heatmap de Correlación":
            ttk.Label(self.plot_specific_controls_frame, text="No requiere selección de columna.").pack(pady=5, anchor="w")


    def _update_plot_controls(self):
        """Actualiza las listas desplegables con las columnas disponibles."""
        self._on_plot_type_change(self.plot_type_var.get()) # Llama a la función para refrescar las listas de columnas

    def _display_plot(self, fig: plt.Figure | None, error_msg: str | None, interpretation: str | None):
        """Muestra el gráfico y su interpretación en la interfaz."""
        # Limpiar el frame anterior del canvas
        for widget in self.plot_canvas_container.winfo_children():
            widget.destroy()

        # Limpiar el texto de interpretación
        self.plot_interpretation_text.config(state=tk.NORMAL)
        self.plot_interpretation_text.delete(1.0, tk.END)
        self.plot_interpretation_text.config(state=tk.DISABLED)

        # Resetear figuras para exportación
        self.current_plot_fig = None
        self.current_plot_interpretation = None
        self.current_plot_title = None

        if fig:
            plot_title_base = f"Gráfico de {self.plot_type_var.get()}"
            if self.plot_type_var.get() == "Diagrama de Dispersión":
                plot_title = f"{plot_title_base}: {self.plot_x_column_var.get()} vs {self.plot_y_column_var.get()}"
            elif self.plot_type_var.get() == "Boxplot Agrupado":
                plot_title = f"{plot_title_base}: {self.plot_numeric_col_var.get()} por {self.plot_category_col_var.get()}"
            elif self.plot_type_var.get() == "Heatmap de Correlación":
                plot_title = plot_title_base
            else: # Histograma, Boxplot, Barras Categóricas
                plot_title = f"{plot_title_base} para {self.plot_column_var.get()}"
            
            self.plot_title_label.config(text=plot_title)
            self.current_plot_title = plot_title # Guardar título
            
            canvas = FigureCanvasTkAgg(fig, master=self.plot_canvas_container)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            canvas.draw()

            # Añadir barra de herramientas de Matplotlib
            toolbar = NavigationToolbar2Tk(canvas, self.plot_canvas_container)
            toolbar.update()
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # Mostrar interpretación
            if interpretation:
                self.plot_interpretation_text.config(state=tk.NORMAL)
                self.plot_interpretation_text.insert(tk.END, interpretation)
                self.plot_interpretation_text.config(state=tk.DISABLED)
                # Configurar tags para negrita si usas Markdown-like
                self.plot_interpretation_text.tag_configure('bold', font=('Segoe UI', 9, 'bold'))
                self._apply_bold_tags(self.plot_interpretation_text) # Función auxiliar para aplicar negrita
            
            self.current_plot_fig = fig # Guardar la figura actual para exportación
            self.current_plot_interpretation = interpretation # Guardar la interpretación

            # No cerrar la figura inmediatamente para permitir la exportación
            # Se cerrará cuando se genere un nuevo plot o se exporte.
        else:
            self.plot_title_label.config(text="Error de Gráfico")
            messagebox.showerror("Error de Gráfico", error_msg)
            # Mostrar error en la interpretación también
            self.plot_interpretation_text.config(state=tk.NORMAL)
            self.plot_interpretation_text.insert(tk.END, f"Error: {error_msg}")
            self.plot_interpretation_text.config(state=tk.DISABLED)
            # Asegurarse de limpiar la figura anterior si hubo un error
            if self.current_plot_fig:
                plt.close(self.current_plot_fig)
                self.current_plot_fig = None

    def _apply_bold_tags(self, text_widget):
        """Aplica formato negrita a texto entre '**' en un widget ScrolledText."""
        text_content = text_widget.get(1.0, tk.END)
        start_idx = 1.0
        while True:
            start_bold = text_widget.search('**', start_idx, stopindex=tk.END)
            if not start_bold:
                break
            
            end_bold = text_widget.search('**', f"{start_bold}+2c", stopindex=tk.END)
            if not end_bold:
                break

            text_widget.tag_add('bold', f"{start_bold}+2c", end_bold)
            
            # Elimina los caracteres '**'
            text_widget.delete(start_bold, f"{start_bold}+2c")
            text_widget.delete(end_bold, f"{end_bold}-2c") # Adjusted for deletion

            # Mueve el índice de inicio para la siguiente búsqueda
            start_idx = text_widget.index(end_bold)

    def generate_plot(self):
        """Genera el gráfico seleccionado."""
        if self.data_handler.get_dataframe() is None:
            messagebox.showerror("Error", "Carga un archivo de datos primero.")
            return
        
        # Cierra la figura anterior si existe para liberar memoria
        if self.current_plot_fig:
            plt.close(self.current_plot_fig)
            self.current_plot_fig = None

        plot_type = self.plot_type_var.get()
        selected_column = self.plot_column_var.get()
        selected_x_column = self.plot_x_column_var.get()
        selected_y_column = self.plot_y_column_var.get()
        selected_numeric_col = self.plot_numeric_col_var.get()
        selected_category_col = self.plot_category_col_var.get()

        df = self.data_handler.get_dataframe()
        fig = None
        error_msg = None
        interpretation_text = None

        try:
            if plot_type == "Histograma":
                fig, error_msg, interpretation_text = self.stats_analyzer.plot_histogram(df, selected_column)
            elif plot_type == "Boxplot":
                fig, error_msg, interpretation_text = self.stats_analyzer.plot_boxplot(df, selected_column)
            elif plot_type == "Diagrama de Dispersión":
                fig, error_msg, interpretation_text = self.stats_analyzer.plot_scatterplot(df, selected_x_column, selected_y_column)
            elif plot_type == "Boxplot Agrupado":
                fig, error_msg, interpretation_text = self.stats_analyzer.plot_grouped_boxplot(df, selected_numeric_col, selected_category_col)
            elif plot_type == "Heatmap de Correlación":
                fig, error_msg, interpretation_text = self.stats_analyzer.plot_correlation_heatmap(df)
            elif plot_type == "Gráfico de Barras (Categórica)":
                fig, error_msg, interpretation_text = self.stats_analyzer.plot_barplot_categorical(df, selected_column)
            else:
                error_msg = "Tipo de gráfico no reconocido."
        except Exception as e:
            error_msg = f"Error inesperado al generar el gráfico: {e}"

        self._display_plot(fig, error_msg, interpretation_text)

    def _export_current_plot_to_pdf(self):
        if self.current_plot_fig is None:
            messagebox.showwarning("Advertencia", "No hay un gráfico generado para exportar.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("Archivos PDF", "*.pdf")]
        )
        if not filepath:
            return

        try:
            doc = SimpleDocTemplate(filepath, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Estilos personalizados para el PDF
            style_title = ParagraphStyle(name='TitleStyle', fontSize=18, leading=22, alignment=TA_CENTER, fontName='Helvetica-Bold')
            style_heading = ParagraphStyle(name='HeadingStyle', fontSize=14, leading=18, alignment=TA_LEFT, fontName='Helvetica-Bold')
            style_body = ParagraphStyle(name='BodyStyle', fontSize=10, leading=14, alignment=TA_LEFT, fontName='Helvetica')
            style_bold = ParagraphStyle(name='BoldStyle', fontSize=10, leading=14, alignment=TA_LEFT, fontName='Helvetica-Bold')


            # Título del documento
            story.append(Paragraph("Reporte de Análisis Gráfico y Estadístico", style_title))
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph(f"Fecha del Reporte: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style_body))
            story.append(Spacer(1, 0.4 * inch))

            # Título del Gráfico
            story.append(Paragraph(f"Gráfico: {self.current_plot_title}", style_heading))
            story.append(Spacer(1, 0.2 * inch))

            # Guardar la figura en un buffer de bytes
            img_buffer = io.BytesIO()
            self.current_plot_fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
            img_buffer.seek(0)
            
            # Ajustar el tamaño de la imagen en el PDF
            img = RLImage(img_buffer)
            img_width, img_height = img.drawWidth, img.drawHeight
            aspect_ratio = img_height / img_width
            
            # Escalar si es muy grande para la página
            max_width = letter[0] - 2 * inch # letter width - margins
            if img_width > max_width:
                img.drawWidth = max_width
                img.drawHeight = max_width * aspect_ratio
            
            story.append(img)
            story.append(Spacer(1, 0.2 * inch))

            # Interpretación del Gráfico
            story.append(Paragraph("Interpretación:", style_heading))
            story.append(Spacer(1, 0.1 * inch))
            
            # Procesar la interpretación para ReportLab (negritas)
            # Reemplazar '**texto**' con '<font name="Helvetica-Bold">texto</font>'
            formatted_interpretation = self.current_plot_interpretation.replace('**', '<font name="Helvetica-Bold">').replace('</font><font name="Helvetica-Bold">', '') + '</font>' if self.current_plot_interpretation else ""
            formatted_interpretation = formatted_interpretation.replace('<font name="Helvetica-Bold">', '<b>').replace('</font>', '</b>') # Usar tags HTML básicos para ReportLab

            story.append(Paragraph(formatted_interpretation, style_body))
            story.append(Spacer(1, 0.4 * inch))

            doc.build(story)
            messagebox.showinfo("Éxito", f"Gráfico y análisis exportados a:\n{filepath}")

            # Cerrar la figura después de exportar para liberar memoria
            plt.close(self.current_plot_fig)
            self.current_plot_fig = None
            self.current_plot_interpretation = None
            self.current_plot_title = None

        except Exception as e:
            messagebox.showerror("Error de Exportación", f"No se pudo exportar a PDF: {e}")
            # En caso de error, también intentar cerrar la figura si estaba abierta
            if self.current_plot_fig:
                plt.close(self.current_plot_fig)
                self.current_plot_fig = None
                self.current_plot_interpretation = None
                self.current_plot_title = None


    # --- Pestaña de Estadística Inferencial ---
    def _create_inferential_tab(self):
        frame = ttk.Frame(self.tab_inferential, padding="20")
        frame.pack(expand=True, fill="both")

        ttk.Label(frame, text="Análisis Estadístico Inferencial", style="Header.TLabel").pack(pady=10)

        # Controles de selección de prueba
        test_control_frame = ttk.Frame(frame)
        test_control_frame.pack(pady=10, fill="x")

        ttk.Label(test_control_frame, text="Selecciona Prueba:").pack(side="left", padx=5)
        self.inferential_test_var = tk.StringVar(value="Prueba t de una Muestra")
        test_types = ["Prueba t de una Muestra", "Prueba t Independiente", "Regresión Lineal Simple", "Regresión Lineal Múltiple", "ANOVA (Un Factor)", "Chi-Cuadrado"]
        ttk.OptionMenu(test_control_frame, self.inferential_test_var, self.inferential_test_var.get(), *test_types, command=self._on_inferential_test_change).pack(side="left", padx=5, expand=True, fill="x")

        # Controles específicos de la prueba
        self.inferential_specific_controls_frame = ttk.Frame(frame)
        self.inferential_specific_controls_frame.pack(pady=10, fill="both", expand=False)
        self._on_inferential_test_change(self.inferential_test_var.get()) # Inicializar

        ttk.Button(frame, text="Ejecutar Prueba", command=self._run_inferential_test).pack(pady=10)

        self.inferential_output_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=20, font=("Courier New", 9), relief="groove")
        self.inferential_output_text.pack(expand=True, fill="both", pady=10)
        self.inferential_output_text.config(state=tk.DISABLED)

    def _on_inferential_test_change(self, selected_test):
        """Actualiza los controles específicos de la prueba inferencial seleccionada."""
        for widget in self.inferential_specific_controls_frame.winfo_children():
            widget.destroy()

        df = self.data_handler.get_dataframe()
        numeric_cols = self.data_handler.get_numeric_columns(df)
        categorical_cols = self.data_handler.get_categorical_columns(df)

        if selected_test == "Prueba t de una Muestra":
            ttk.Label(self.inferential_specific_controls_frame, text="Columna Numérica:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            self.one_sample_t_col_var = tk.StringVar()
            self.one_sample_t_col_menu = ttk.OptionMenu(self.inferential_specific_controls_frame, self.one_sample_t_col_var, "", *numeric_cols)
            self.one_sample_t_col_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
            self._update_column_listbox(dropdown_menu=self.one_sample_t_col_menu, string_var=self.one_sample_t_col_var, column_type="numeric")

            ttk.Label(self.inferential_specific_controls_frame, text="Media Poblacional Hipotetizada:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
            self.pop_mean_entry = ttk.Entry(self.inferential_specific_controls_frame)
            self.pop_mean_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
            self.pop_mean_entry.insert(0, "0.0") # Valor por defecto

        elif selected_test == "Prueba t Independiente":
            ttk.Label(self.inferential_specific_controls_frame, text="Columna Numérica (Dependiente):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            self.indep_t_num_col_var = tk.StringVar()
            self.indep_t_num_col_menu = ttk.OptionMenu(self.inferential_specific_controls_frame, self.indep_t_num_col_var, "", *numeric_cols)
            self.indep_t_num_col_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
            self._update_column_listbox(dropdown_menu=self.indep_t_num_col_menu, string_var=self.indep_t_num_col_var, column_type="numeric")

            ttk.Label(self.inferential_specific_controls_frame, text="Columna Categórica (Grupo - 2 categorías):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
            self.indep_t_group_col_var = tk.StringVar()
            self.indep_t_group_col_menu = ttk.OptionMenu(self.inferential_specific_controls_frame, self.indep_t_group_col_var, "", *categorical_cols)
            self.indep_t_group_col_menu.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
            self._update_column_listbox(dropdown_menu=self.indep_t_group_col_menu, string_var=self.indep_t_group_col_var, column_type="categorical")

        elif selected_test == "Regresión Lineal Simple":
            ttk.Label(self.inferential_specific_controls_frame, text="Variable Dependienxte (Y):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            self.slr_dep_var = tk.StringVar()
            self.slr_dep_menu = ttk.OptionMenu(self.inferential_specific_controls_frame, self.slr_dep_var, "", *numeric_cols)
            self.slr_dep_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
            self._update_column_listbox(dropdown_menu=self.slr_dep_menu, string_var=self.slr_dep_var, column_type="numeric")

            ttk.Label(self.inferential_specific_controls_frame, text="Variable Independiente (X):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
            self.slr_indep_var = tk.StringVar()
            self.slr_indep_menu = ttk.OptionMenu(self.inferential_specific_controls_frame, self.slr_indep_var, "", *numeric_cols)
            self.slr_indep_menu.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
            self._update_column_listbox(dropdown_menu=self.slr_indep_menu, string_var=self.slr_indep_var, column_type="numeric")

        elif selected_test == "Regresión Lineal Múltiple":
            ttk.Label(self.inferential_specific_controls_frame, text="Variable Dependiente (Y):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            self.mlr_dep_var = tk.StringVar()
            self.mlr_dep_menu = ttk.OptionMenu(self.inferential_specific_controls_frame, self.mlr_dep_var, "", *numeric_cols)
            self.mlr_dep_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
            self._update_column_listbox(dropdown_menu=self.mlr_dep_menu, string_var=self.mlr_dep_var, column_type="numeric")

            ttk.Label(self.inferential_specific_controls_frame, text="Variables Independientes (X):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
            self.mlr_indep_listbox = tk.Listbox(self.inferential_specific_controls_frame, selectmode=tk.MULTIPLE, height=5, exportselection=0, font=("Segoe UI", 9))
            self.mlr_indep_listbox.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
            mlr_scrollbar = ttk.Scrollbar(self.inferential_specific_controls_frame, orient="vertical", command=self.mlr_indep_listbox.yview)
            mlr_scrollbar.grid(row=2, column=2, sticky="ns")
            self.mlr_indep_listbox.config(yscrollcommand=mlr_scrollbar.set)
            self._update_column_listbox(listbox_widget=self.mlr_indep_listbox, column_type="numeric")

        elif selected_test == "ANOVA (Un Factor)":
            ttk.Label(self.inferential_specific_controls_frame, text="Columna Numérica (Dependiente):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            self.anova_num_col_var = tk.StringVar()
            self.anova_num_col_menu = ttk.OptionMenu(self.inferential_specific_controls_frame, self.anova_num_col_var, "", *numeric_cols)
            self.anova_num_col_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
            self._update_column_listbox(dropdown_menu=self.anova_num_col_menu, string_var=self.anova_num_col_var, column_type="numeric")

            ttk.Label(self.inferential_specific_controls_frame, text="Columna Categórica (Factor de Grupo):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
            self.anova_group_col_var = tk.StringVar()
            self.anova_group_col_menu = ttk.OptionMenu(self.inferential_specific_controls_frame, self.anova_group_col_var, "", *categorical_cols)
            self.anova_group_col_menu.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
            self._update_column_listbox(dropdown_menu=self.anova_group_col_menu, string_var=self.anova_group_col_var, column_type="categorical")

        elif selected_test == "Chi-Cuadrado":
            ttk.Label(self.inferential_specific_controls_frame, text="Columna Categórica 1:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            self.chi2_col1_var = tk.StringVar()
            self.chi2_col1_menu = ttk.OptionMenu(self.inferential_specific_controls_frame, self.chi2_col1_var, "", *categorical_cols)
            self.chi2_col1_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
            self._update_column_listbox(dropdown_menu=self.chi2_col1_menu, string_var=self.chi2_col1_var, column_type="categorical")

            ttk.Label(self.inferential_specific_controls_frame, text="Columna Categórica 2:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
            self.chi2_col2_var = tk.StringVar()
            self.chi2_col2_menu = ttk.OptionMenu(self.inferential_specific_controls_frame, self.chi2_col2_var, "", *categorical_cols)
            self.chi2_col2_menu.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
            self._update_column_listbox(dropdown_menu=self.chi2_col2_menu, string_var=self.chi2_col2_var, column_type="categorical")

        self.inferential_specific_controls_frame.grid_columnconfigure(1, weight=1)

    def _update_inferential_controls(self):
        """Refresca los controles de las pruebas inferenciales."""
        self._on_inferential_test_change(self.inferential_test_var.get())

    def _run_inferential_test(self):
        if self.data_handler.get_dataframe() is None:
            messagebox.showwarning("Advertencia", "Carga un archivo de datos primero.")
            return

        test_type = self.inferential_test_var.get()
        df = self.data_handler.get_dataframe()
        output = "Error: Parámetros inválidos o prueba no implementada."

        try:
            if test_type == "Prueba t de una Muestra":
                col = self.one_sample_t_col_var.get()
                pop_mean_str = self.pop_mean_entry.get()
                if not col or not pop_mean_str:
                    messagebox.showwarning("Advertencia", "Selecciona una columna y una media poblacional hipotetizada.")
                    return
                try:
                    pop_mean = float(pop_mean_str)
                except ValueError:
                    messagebox.showerror("Error de Entrada", "La media poblacional debe ser un número.")
                    return
                output = self.stats_analyzer.run_one_sample_t_test(df, col, pop_mean)
            
            elif test_type == "Prueba t Independiente":
                num_col = self.indep_t_num_col_var.get()
                group_col = self.indep_t_group_col_var.get()
                if not num_col or not group_col:
                    messagebox.showwarning("Advertencia", "Selecciona una columna numérica y una de grupo.")
                    return
                output = self.stats_analyzer.run_independent_t_test(df, num_col, group_col)

            elif test_type == "Regresión Lineal Simple":
                dep_var = self.slr_dep_var.get()
                indep_var = self.slr_indep_var.get()
                if not dep_var or not indep_var:
                    messagebox.showwarning("Advertencia", "Selecciona variables dependiente e independiente.")
                    return
                output = self.stats_analyzer.run_simple_linear_regression(df, dep_var, indep_var)

            elif test_type == "Regresión Lineal Múltiple":
                dep_var = self.mlr_dep_var.get()
                selected_indices = self.mlr_indep_listbox.curselection()
                indep_vars = [self.mlr_indep_listbox.get(i) for i in selected_indices]
                if not dep_var or not indep_vars:
                    messagebox.showwarning("Advertencia", "Selecciona una variable dependiente y al menos una independiente.")
                    return
                output = self.stats_analyzer.run_multiple_linear_regression(df, dep_var, indep_vars)

            elif test_type == "ANOVA (Un Factor)":
                num_col = self.anova_num_col_var.get()
                group_col = self.anova_group_col_var.get()
                if not num_col or not group_col:
                    messagebox.showwarning("Advertencia", "Selecciona una columna numérica y una de grupo.")
                    return
                output = self.stats_analyzer.run_anova(df, num_col, group_col)

            elif test_type == "Chi-Cuadrado":
                col1 = self.chi2_col1_var.get()
                col2 = self.chi2_col2_var.get()
                if not col1 or not col2:
                    messagebox.showwarning("Advertencia", "Selecciona dos columnas categóricas.")
                    return
                if col1 == col2:
                    messagebox.showwarning("Advertencia", "Las columnas para Chi-Cuadrado deben ser diferentes.")
                    return
                output = self.stats_analyzer.run_chi_squared_test(df, col1, col2)

        except Exception as e:
            output = f"Ocurrió un error al ejecutar la prueba: {e}"

        self.inferential_output_text.config(state=tk.NORMAL)
        self.inferential_output_text.delete(1.0, tk.END)
        self.inferential_output_text.insert(tk.END, output)
        self.inferential_output_text.config(state=tk.DISABLED)

    # --- Pestaña de Meta-Análisis / Resumen Ejecutivo ---
    def _create_meta_analysis_tab(self):
        frame = ttk.Frame(self.tab_meta_analysis, padding="20")
        frame.pack(expand=True, fill="both")

        ttk.Label(frame, text="Resumen Ejecutivo de Análisis", style="Header.TLabel").pack(pady=10)

        info_label = ttk.Label(frame, text="Aquí se presenta un resumen de todos los análisis y visualizaciones realizados en la sesión actual.", style="Info.TLabel")
        info_label.pack(pady=5)

        self.meta_analysis_output_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=25, font=("Segoe UI", 9), relief="groove")
        self.meta_analysis_output_text.pack(expand=True, fill="both", pady=10)
        self.meta_analysis_output_text.config(state=tk.DISABLED)

        ttk.Button(frame, text="Generar Resumen Completo PDF", command=self._export_full_summary_to_pdf).pack(pady=10)

    def _update_meta_analysis_content(self):
        """Actualiza el contenido del resumen ejecutivo."""
        history = self.data_handler.get_analysis_history()
        
        self.meta_analysis_output_text.config(state=tk.NORMAL)
        self.meta_analysis_output_text.delete(1.0, tk.END)

        if not history:
            self.meta_analysis_output_text.insert(tk.END, "Aún no se han realizado análisis en esta sesión. Realice algunas pruebas o visualizaciones para ver el resumen aquí.")
        else:
            self.meta_analysis_output_text.insert(tk.END, "--- Historial de Análisis de la Sesión Actual ---\n\n")
            for entry in history:
                self.meta_analysis_output_text.insert(tk.END, f"[{entry['timestamp']}] - **{entry['type']}**\n", 'bold_tag')
                self.meta_analysis_output_text.insert(tk.END, f"  Descripción: {entry['description']}\n")
                if entry['result_summary']:
                    self.meta_analysis_output_text.insert(tk.END, f"  Resumen de Resultados: {entry['result_summary']}\n")
                self.meta_analysis_output_text.insert(tk.END, "--------------------------------------------------\n\n")
            
            self.meta_analysis_output_text.tag_configure('bold_tag', font=('Segoe UI', 9, 'bold'))

        self.meta_analysis_output_text.config(state=tk.DISABLED)

    def _export_full_summary_to_pdf(self):
        """Exporta el resumen completo de análisis a un PDF."""
        history = self.data_handler.get_analysis_history()
        if not history:
            messagebox.showwarning("Advertencia", "No hay historial de análisis para exportar a PDF.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("Archivos PDF", "*.pdf")]
        )
        if not filepath:
            return

        try:
            doc = SimpleDocTemplate(filepath, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Estilos personalizados para el PDF
            style_title = ParagraphStyle(name='TitleStyle', fontSize=18, leading=22, alignment=TA_CENTER, fontName='Helvetica-Bold')
            style_h1 = ParagraphStyle(name='H1Style', fontSize=16, leading=20, alignment=TA_LEFT, fontName='Helvetica-Bold')
            style_h2 = ParagraphStyle(name='H2Style', fontSize=12, leading=16, alignment=TA_LEFT, fontName='Helvetica-Bold')
            style_body = ParagraphStyle(name='BodyStyle', fontSize=10, leading=14, alignment=TA_LEFT, fontName='Helvetica')
            style_code = ParagraphStyle(name='CodeStyle', fontSize=9, leading=12, alignment=TA_LEFT, fontName='Courier', backColor='#e0e0e0', borderWidth=0.5, borderColor='#9e9e9e', borderPadding=5)

            # Título del Reporte
            story.append(Paragraph("Reporte de Resumen de Análisis Estadístico", style_title))
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph(f"Fecha del Reporte: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style_body))
            if self.data_handler.filepath:
                story.append(Paragraph(f"Dataset Analizado: {os.path.basename(self.data_handler.filepath)}", style_body))
            story.append(Spacer(1, 0.4 * inch))

            story.append(Paragraph("Historial Detallado de Análisis:", style_h1))
            story.append(Spacer(1, 0.2 * inch))

            for entry in history:
                story.append(Paragraph(f"<b>[{entry['timestamp']}] - {entry['type']}</b>", style_body))
                story.append(Paragraph(f"<b>Descripción:</b> {entry['description']}", style_body))
                if entry['result_summary']:
                    story.append(Paragraph(f"<b>Resumen de Resultados:</b> {entry['result_summary']}", style_body))
                story.append(Spacer(1, 0.1 * inch))
                story.append(Paragraph("-" * 60, style_body)) # Separador visual
                story.append(Spacer(1, 0.2 * inch))

            doc.build(story)
            messagebox.showinfo("Éxito", f"Resumen de análisis exportado a:\n{filepath}")

        except Exception as e:
            messagebox.showerror("Error de Exportación", f"No se pudo exportar el resumen a PDF: {e}")


if __name__ == "__main__":
    app = ProStatApp()
    app.mainloop()