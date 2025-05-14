from PIL import Image
import io

def is_valid_image(image_data: bytes) -> bool:
    """Fungsi untuk memeriksa apakah data gambar valid"""
    try:
        image = Image.open(io.BytesIO(image_data))
        image.verify()  # Memastikan file adalah gambar yang valid
        return True
    except Exception:
        return False
