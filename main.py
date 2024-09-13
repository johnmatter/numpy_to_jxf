import numpy as np
import os
import datetime
import struct
import unittest
import pdb
from PIL import Image

def export_to_jxf(matrix: np.ndarray, filename: str = None) -> None:
    """
    Export a NumPy ndarray to a Max/MSP binary Jitter matrix format (.jxf).

    Parameters:
    - matrix: NumPy ndarray to export.
    - filename: Output filename for the .jxf file.
    """

    # Determine the matrix type and coerce if necessary
    if matrix.dtype == np.uint8:
        matrix_type = b'CHAR'
    elif matrix.dtype == np.float32:
        matrix_type = b'FL32'
    elif matrix.dtype == np.float64:
        matrix_type = b'FL64'
    elif matrix.dtype.kind == 'i':  # For int32
        matrix = matrix.astype(np.int32)
        matrix_type = b'LONG'
    else:
        raise ValueError(f"Unsupported data type: {matrix.dtype}")

    # Create filename if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}.jxf"

    if matrix.ndim == 3 and matrix.shape[2] in [3, 4]:
        # Reshape 3D color image to 2D with 4 planes
        if matrix.shape[2] == 3:
            matrix = np.pad(matrix, ((0,0), (0,0), (0,1)), mode='constant', constant_values=255)
        matrix = matrix.reshape(matrix.shape[0], -1)
        dim_count = 2
        plane_count = 4
        dimensions = matrix.shape[::-1]
    else:
        dim_count = matrix.ndim
        dimensions = matrix.shape[::-1]
        plane_count = 1

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

        # Write matrix data
        f.write(matrix.tobytes())

class TestExportToJXF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image_filename = "pete.png"
        cls.image = read_and_downscale_image(cls.image_filename, (100, 133))

    def test_export_uint8(self):
        export_to_jxf(self.image.astype(np.uint8), 'test_uint8.jxf')
        self.assertTrue(os.path.exists('test_uint8.jxf'))

    def test_export_float32(self):
        export_to_jxf(self.image.astype(np.float32), 'test_float32.jxf')
        self.assertTrue(os.path.exists('test_float32.jxf'))

    def test_export_float64(self):
        export_to_jxf(self.image.astype(np.float64), 'test_float64.jxf')
        self.assertTrue(os.path.exists('test_float64.jxf'))

    def test_export_3d_uint8(self):
        export_to_jxf(self.image.astype(np.uint8), 'test_3d_uint8.jxf')
        self.assertTrue(os.path.exists('test_3d_uint8.jxf'))

    def test_export_3d_float32(self):
        export_to_jxf(self.image.astype(np.float32), 'test_3d_float32.jxf')
        self.assertTrue(os.path.exists('test_3d_float32.jxf'))

    def test_export_3d_float64(self):
        export_to_jxf(self.image.astype(np.float64), 'test_3d_float64.jxf')
        self.assertTrue(os.path.exists('test_3d_float64.jxf'))

    def test_export_invalid_dtype(self):
        with self.assertRaises(ValueError):
            export_to_jxf(np.array([[1, 2], [3, 4]], dtype=np.int16))

def read_and_downscale_image(filename: str, resolution: tuple) -> np.ndarray:
  """
  Read an image from a file and downscale it to the specified resolution.

  Parameters:
  - filename: The path to the image file.
  - resolution: A tuple specifying the desired width and height of the output image.

  Returns:
  - A NumPy ndarray representing the downscaled image.
  """
  # Open the image file
  with Image.open(filename) as img:
    # Downscale the image
    img_resized = img.resize(resolution, Image.LANCZOS)
    
  # Convert the image to a NumPy array
  image_array = np.array(img_resized)

  # Normalize the image data
  if image_array.dtype == np.uint8:
      image_array = image_array.astype(np.float32) / 255.0

  return image_array


if __name__ == '__main__':
    unittest.main()