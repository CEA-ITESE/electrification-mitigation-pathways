#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Any, Optional, Literal, Union, ClassVar, Final
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from pydantic import model_validator, validate_call
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

from itese_dataviz.schemas import SourceDatasetSchema, ReferenceDatasetSchema

class BasePlot(BaseModel, ABC):
    """Abstract base class acting as the foundation for all plot engines.
    
    Handles core Pydantic configuration, generic DataFrame properties,
    and automated structural validations via Pandera workflows.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core shared fields across any chart type in the package
    data: pd.DataFrame = Field(description="The study dataset.")
    reference_data: Optional[pd.DataFrame] = Field(
        default=None,
        description="Optional reference dataset to overlay (e.g., AR6 database)."
    )
    index_col: list[str] = Field(description="List of columns to use as identifiers")
    display: dict[str, str | list[int]] = Field(default_factory=None)
    options: dict[str, str | list[int] | bool] = Field(default_factory=dict)

    # Optional fields
    # col_filters: list[str] = Field(default_factory=list)
    periods: list[int] = Field(default_factory=list)
    row_filters: dict[str, list[str | int]] = Field(default_factory=dict)
    region_map: dict[str, str] = Field(default_factory=dict)
    techno_map: dict[str, str] = Field(default_factory=dict)
    categorical: dict[str, list[str | int]] = Field(default_factory=dict)
    col_names: dict[str, list[str]] | None = Field(default=None)
    nan_values: list | str | int | float | None = Field(default=None)
    reference_index_col: list[str] | None = Field(default=None)
    reference_map_col: dict[str, str] = Field(default_factory=dict)
    reference_row_filters: dict[str, list[str | int]] | None = Field(default=None)
    reference_col_names: dict[str, list[str]] = Field(default_factory=dict)
    reference_nan_values: list | str | int | float | None = Field(default=None)

    _col_filters: list[str] = PrivateAttr(default_factory=set)
    _fig: Optional[sns.FacetGrid] = PrivateAttr(default=None)

    _REQUIRED_KEYS: Final[tuple[str, ...]] = (
        "Grid",
        "ColorStrip",
        "Yaxis",
    )

    _ALLOWED_OPTIONS: Final[tuple[str, ...]] = ()

    @model_validator(mode="after")
    def preprocess_pipeline(self) -> "BasePlot":
        """Generic execution pipeline validating inputs before rendering begins."""
        # Enforce strict schema validation on the primary dataset
        self.data = SourceDatasetSchema.validate(self.data)

        # Enforce schema validation on reference records only if supplied
        if self.reference_data is not None:
            self.reference_data = ReferenceDatasetSchema.validate(self.reference_data)

        if self.display:
            missing_keys = set(self._REQUIRED_KEYS) - self.display.keys()
            if missing_keys:
                raise KeyError(f"'display' argument missing mandatory keys : {missing_keys}")

        if self.options:
            invalid_keys = set(self.options.keys()) - set(self._ALLOWED_OPTIONS)
            if invalid_keys:
                raise KeyError(f"'options' argument contains invalid keys : {invalid_keys}")

            # assign options to instance attributes for direct access
            if self.options:
                for key, value in self.options.items():
                    setattr(self, key, value)

        return self

    @model_validator(mode="after")
    def preprocess_reference_data(self) -> "BasePlot":
        if self.reference_data is not None and self.reference_index_col is None:
            raise ValueError(
                "Field 'reference_index_col' is mandatory when 'reference_data' is provided."
            )

        if self.reference_data is not None and not self.reference_col_names:
            raise ValueError(
                "Field 'reference_map_col' is mandatory when 'reference_data' is provided."
            )

        return self

    def _melt_indicators(self) -> None:
        """
        Melt (unpivot) the DataFrame to long format for plotting, using index_col as identifiers.
        """

        if self.index_col:
            # TODO change names to variables
            self.data = self.data.melt(
                id_vars=self.index_col,
                var_name="Technology",
                value_name=self.display['Yaxis']
            )

        if self.reference_index_col:
            # TODO change names to variables
            self.reference_data = self.reference_data.melt(
                id_vars=self.reference_index_col,
                var_name="Technology",
                value_name=self.display['Yaxis']
            )

    @staticmethod
    def _replace_na(df: pd.DataFrame, targets: list, tolerance: float = 1e-5):

        work_df = df.copy()

        exact_targets = []
        numeric_targets = []

        for val in targets:
            # Attention: en Python, isinstance(True, int) est True, on exclut donc les boooléens
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                numeric_targets.append(float(val))
            else:
                exact_targets.append(val)

        if exact_targets:
            work_df = work_df.replace(exact_targets, pd.NA)

        if numeric_targets:
            for col in work_df.columns:
                if pd.api.types.is_numeric_dtype(work_df[col]):
                    s_numeric = work_df[col]

                else:
                    s_converted = pd.to_numeric(work_df[col], errors="coerce")
                    if s_converted.notna().any():
                        # string but convertible
                        work_df[col] = s_converted
                        s_numeric = s_converted
                    else:
                        # not convertible
                        continue

                # replace na with tolerance
                for num_val in numeric_targets:
                    is_close_mask = (s_numeric - num_val).abs() < tolerance
                    work_df[col] = work_df[col].mask(is_close_mask, pd.NA)

        return work_df

    def _drop_na(self):

        self.data = self.data.dropna()
        for col in self.data.columns:
            if pd.api.types.is_object_dtype(self.data[col]):
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce")

        self.reference_data = self.reference_data.dropna()
        if not self.reference_data is None:
            for col in self.reference_data:
                if pd.api.types.is_object_dtype(self.reference_data[col]):
                    self.reference_data[col] = pd.to_numeric(self.reference_data[col], errors="coerce")


    def _convert_categorical(self) -> None:
        """
        Convert specified columns to categorical types with defined order.
        """
        for col, categories in self.categorical.items():
            self.data[col] = pd.Categorical(self.data[col], categories=categories, ordered=True)

        # TODO change as variable column
        if not self.reference_data is None:

            self.reference_data["Technology"] = pd.Categorical(self.reference_data["Technology"], categories=self._col_filters, ordered=True)

    @validate_call
    def _aggregate_columns(self,
        mapping: dict[str, list[str]],
        target: Literal["source", "reference"],
        aggregate: Literal["sum", "mean", "min", "max"],
        drop_original: bool = False
        ):

        # TODO drop_original not implemented

        target_df = self.data if target == "source" else self.reference_data

        # TODO replace if cases with getattr to dynamically access the target DataFrame based on the 'target' argument
        for new_col, cols_to_aggregate in mapping.items():
            if aggregate == "sum":
                target_df[new_col] = target_df[cols_to_aggregate].sum(axis=1)
            elif aggregate == "mean":
                target_df[new_col] = target_df[cols_to_aggregate].mean(axis=1)
            elif aggregate == "min":
                target_df[new_col] = target_df[cols_to_aggregate].min(axis=1)
            elif aggregate == "max":
                target_df[new_col] = target_df[cols_to_aggregate].max(axis=1)
            else:
                raise ValueError(f"Unsupported aggregation method: {aggregate}")


        if target == "source":
            self.data = target_df
        elif target == "reference":
            self.reference_data = target_df


    @abstractmethod
    def plot(self, **kwargs: Any) -> Any:
        """Render the compiled visualization layout.
        
        This contract interface must be implemented by all subclasses.
        """
        pass


    def export(
        self,
        filename: Union[str, Path],
        dpi: int = 400,
        bbox_inches: str = "tight",
        pad_inches: float = 0.1,
        transparent: bool = False,
    ) -> None:
        """Sauvegarde le graphique généré dans un fichier (PNG, PDF, SVG, etc.).

        Args:
            filename: Chemin du fichier de sortie (ex: 'plots/output.png').
            dpi: Résolution de l'image (300 recommandé pour la qualité).
            bbox_inches: 'tight' pour éviter de couper les titres/légendes.
            pad_inches: Marge autour de la figure.
            transparent: Fond transparent ou non.
        """
        if self._fig is None:
            raise RuntimeError(
                "Le graphique n'a pas encore été généré. "
                "Veuillez exécuter .plot() avant d'appeler .save()."
            )

        # Création automatique des dossiers s'ils n'existent pas
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Sauvegarde via le FacetGrid
        self._fig.savefig(
            filepath,
            dpi=dpi,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            transparent=transparent,
        )
        print(f"Graphique sauvegardé : {filepath.resolve()}")

    def show(self):
        plt.show()