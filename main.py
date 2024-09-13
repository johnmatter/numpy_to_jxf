import numpy as np
import os
import datetime
import struct
import unittest

def export_to_jxf(matrix: np.ndarray, filename: str = None) -> None:
  """
  Export a NumPy ndarray to a Max/MSP binary Jitter matrix format (.jxf).

  Parameters:
  - matrix: NumPy ndarray to export.
  - filename: Output filename for the .jxf file.
  """

  # Coerce ints, floats, and chars if needed
  if matrix.dtype.kind in ['i', 'u', 'b'] and matrix.dtype != np.uint8:
      matrix = matrix.astype(np.uint8)
      matrix_type = b'CHAR'
  elif matrix.dtype.kind == 'f' and matrix.dtype != np.float32:
      matrix = matrix.astype(np.float32)
      matrix_type = b'FL32'
  elif matrix.dtype.kind == 'S':
      matrix = np.frombuffer(matrix.tobytes(), dtype=np.uint8).reshape(matrix.shape)
      matrix_type = b'CHAR'
  elif matrix.dtype == np.uint8:  # Correctly handle uint8
      matrix_type = b'CHAR'
  elif matrix.dtype == np.float32:  # Correctly handle float32
      matrix_type = b'FL32'
  elif matrix.dtype.kind == 'i':  # For int32
      matrix = matrix.astype(np.int32)
      matrix_type = b'LONG'
  elif matrix.dtype == np.float64:
      matrix = matrix.astype(np.float64)
      matrix_type = b'FL64'
  else:
      raise ValueError(f"Unsupported data type: {matrix.dtype}")

  # Check if the data type is supported (float32 or uint8)
  if matrix.dtype not in [np.float32, np.uint8]:
      raise ValueError(f"Data type must be float32 or uint8. Got {matrix.dtype}")

  # Create filename
  if filename is None:
      timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
      filename = f"{timestamp}.jxf"

  # Prepare header information
  match matrix.dtype:
      case np.uint8:
          matrix_type = b'CHAR'
      case np.float32:
          matrix_type = b'FL32'
      case _:
          raise ValueError(f"Unsupported data type: {matrix.dtype}")
  plane_count = 1 if matrix.ndim == 2 else matrix.shape[2]
  dim_count = 2 if matrix.ndim == 2 else 3
  dimensions = matrix.shape[:2][::-1]  # Reverse for row-major order

  # Calculate sizes
  matrix_data_size = matrix.nbytes
  matrix_header_size = 24 + (4 * dim_count)
  matrix_chunk_size = matrix_header_size + matrix_data_size
  total_file_size = 20 + 20 + matrix_chunk_size  # Container + Format + Matrix chunks

  # Open the file for writing in binary mode
  with open(filename, 'wb') as f:
      # Write Container Chunk
      f.write(b'FORM')
      f.write(struct.pack('>I', total_file_size - 8))
      f.write(b'JIT!')

      # Write Format Chunk
      f.write(b'FVER')
      f.write(struct.pack('>I', 12))
      f.write(struct.pack('>I', 0x3C93DC80))  # JIT_BIN_VERSION_1

      # Write Matrix Chunk
      f.write(b'MTRX')
      f.write(struct.pack('>I', matrix_chunk_size))
      f.write(struct.pack('>I', matrix_header_size))
      f.write(matrix_type)
      f.write(struct.pack('>I', plane_count))
      f.write(struct.pack('>I', dim_count))
      for dim in dimensions:
          f.write(struct.pack('>I', dim))

      # Write matrix data based on type
      if matrix_type == b'CHAR':
          f.write(matrix.tobytes())
      elif matrix_type == b'LONG':
          f.write(matrix.astype('>i4').tobytes())
      elif matrix_type == b'FL32':
          f.write(matrix.astype('>f4').tobytes())
      elif matrix_type == b'FL64':
          f.write(matrix.astype('>f8').tobytes())

class TestExportToJXF(unittest.TestCase):

    def test_export_uint8(self):
        matrix = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        export_to_jxf(matrix, 'test_uint8.jxf')
        # Add assertions to verify the output file

    def test_export_float32(self):
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        export_to_jxf(matrix, 'test_float32.jxf')
        # Add assertions to verify the output file

    def test_export_float64(self):
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        export_to_jxf(matrix, 'test_float64.jxf')
        # Add assertions to verify the output file

    def test_export_3d_uint8(self):
        matrix = np.random.randint(0, 256, (2, 2, 2), dtype=np.uint8)
        export_to_jxf(matrix, 'test_3d_uint8.jxf')
        # Add assertions to verify the output file

    def test_export_3d_float32(self):
        matrix = np.random.rand(2, 2, 2).astype(np.float32)
        export_to_jxf(matrix, 'test_3d_float32.jxf')
        # Add assertions to verify the output file

    def test_export_3d_float64(self):
        matrix = np.random.rand(2, 2, 2).astype(np.float64)
        export_to_jxf(matrix, 'test_3d_float64.jxf')
        # Add assertions to verify the output file

    def test_export_invalid_dtype(self):
        matrix = np.array([[1, 2], [3, 4]], dtype=np.int32)
        with self.assertRaises(ValueError):
            export_to_jxf(matrix)

if __name__ == '__main__':
    unittest.main()