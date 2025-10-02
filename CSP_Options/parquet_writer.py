import numpy as np
from csp.adapters.output_adapters.parquet import (
    ParquetOutputConfig,
    ParquetWriter,
    resolve_array_shape_column_name,
)
from csp.adapters.output_adapters.parquet_utility_nodes import flatten_numpy_array
from csp.impl.types.tstype import ts
from csp.impl.types.typing_utils import CspTypingUtils

import csp


@csp.node
def extract_array_element(array_ts: ts[np.ndarray], index: int) -> ts[float]:
    """Extract a single element from a numpy array time series."""
    if csp.ticked(array_ts):
        return float(array_ts[index])


class GregsParquetWriter(ParquetWriter):
    """
    A Parquet writer that optionally omits the extra 'dimensions' column for
    multi-dimensional arrays, controlled by the `include_dimensions` flag.
    """

    def __init__(
        self,
        file_name: str,
        timestamp_column_name,
        config: ParquetOutputConfig = None,
        filename_provider: ts[str] = None,
        split_columns_to_files: bool = False,
        file_metadata=None,
        column_metadata=None,
        file_visitor=None,
        include_dimensions: bool = True,
    ):
        """
        :param include_dimensions: If True (default), write an extra column for the shape
                                   of any multi-dimensional array; if False, omit it.
        """
        super().__init__(
            file_name=file_name,
            timestamp_column_name=timestamp_column_name,
            config=config,
            filename_provider=filename_provider,
            split_columns_to_files=split_columns_to_files,
            file_metadata=file_metadata,
            column_metadata=column_metadata,
            file_visitor=file_visitor,
        )
        self.include_dimensions = include_dimensions

    def publish(self, column_name, value: ts[object], array_dimensions_column_name=None):
        """
        Publish a time series to the Parquet/Arrow file, optionally skipping
        the '_csp_dimensions' column if include_dimensions=False.
        """
        ts_type = value.tstype.typ

        # Check if this is a Numpy array time series.
        if CspTypingUtils.is_numpy_array_type(ts_type):
            # Flatten NumPy array and its shape.
            flat_value, shape_series = flatten_numpy_array(value)._values()

            if self.include_dimensions:
                # Use CSP's default approach: store an additional column for shape.
                shape_col_name = resolve_array_shape_column_name(
                    column_name,
                    array_dimensions_column_name,
                )
                super().publish(shape_col_name, shape_series)

            # Always write the flattened data (1D array).
            return super().publish(column_name, flat_value)

        # Otherwise, for non-array data, just fall back to the parent class's publish.
        return super().publish(column_name, value, array_dimensions_column_name)

    def publish_vector_as_columns(
        self, base_column_name: str, vector_ts: ts[np.ndarray], vector_length: int
    ):
        """
        Publish a vector time series as separate columns, one for each element.

        :param base_column_name: Base name for columns (e.g., 'param' -> 'param_0', 'param_1', etc.)
        :param vector_ts: Time series containing numpy arrays/vectors
        :param vector_length: Expected length of the vector (number of elements)
        """
        for i in range(vector_length):
            element_ts = extract_array_element(vector_ts, i)
            column_name = f"{base_column_name}_{i}"
            self.publish(column_name, element_ts)




