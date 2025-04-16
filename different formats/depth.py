import os
import subprocess
from PIL import Image
import base64

# File paths (adjust these as needed)
image_path = "5_left.png"        # Your original image (PNG)
depth_path = "5_depth.png"             # Your depth map (PNG)
output_path = "output_with_depth.jpg"  # Final JPEG output

# Step 1: Convert the PNG image to JPEG
image = Image.open(image_path).convert("RGB")
image.save(output_path, quality=95)

# Step 2: Open and resize the depth map to reduce file size
depth_img = Image.open(depth_path)
max_width = 640  # Maximum width to reduce file size (adjust as needed)
if depth_img.width > max_width:
    ratio = max_width / depth_img.width
    new_height = int(depth_img.height * ratio)
    depth_img = depth_img.resize((max_width, new_height), Image.LANCZOS)

# Optional: Save the resized depth map temporarily as JPEG with compression
temp_depth_path = "temp_depth.jpg"
depth_img.save(temp_depth_path, quality=50)  # Lower quality for further compression

# Step 3: Encode the (resized and compressed) depth map as Base64
with open(temp_depth_path, "rb") as depth_file:
    depth_base64 = base64.b64encode(depth_file.read()).decode("utf-8")

# Remove the temporary depth file
os.remove(temp_depth_path)

# Step 4: Create the XMP metadata content using Facebook Depth Map format
xmp_template = f"""<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x='adobe:ns:meta/' x:xmptk='XMP Core 5.4.0'>
   <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
      <rdf:Description rdf:about=''
         xmlns:GDepth='http://ns.google.com/photos/1.0/depthmap/'>
         <GDepth:Format>RangeLinear</GDepth:Format>
         <GDepth:Near>0.1</GDepth:Near>
         <GDepth:Far>10.0</GDepth:Far>
         <GDepth:Data>{depth_base64}</GDepth:Data>
      </rdf:Description>
   </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>"""

# Write the XMP metadata to a temporary file
xmp_filename = "temp_metadata.xmp"
with open(xmp_filename, "w", encoding="utf-8") as f:
    f.write(xmp_template)

# Step 5: Use exiftool to embed the XMP metadata into the JPEG image
cmd = ["exiftool", "-overwrite_original", "-XMP<=temp_metadata.xmp", output_path]
subprocess.run(cmd, check=True)

# Remove the temporary XMP file
os.remove(xmp_filename)

print(f"✅ Facebook Depth Map saved as: {output_path}")
