import subprocess

def create_stereo_xmp(stereo_mode, xmp_file_path):
    """
    Creates an XMP file that marks an image as a stereoscopic (SBS) image.
    
    Args:
        stereo_mode (str): A string describing the stereo mode, e.g., "SBS".
        xmp_file_path (str): The output path for the XMP file.
    """
    xmp_content = f'''<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:Stereo="http://ns.adobe.com/stereoscopic/1.0/">
      <Stereo:StereoMode>{stereo_mode}</Stereo:StereoMode>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>'''
    with open(xmp_file_path, "w", encoding="utf-8") as f:
        f.write(xmp_content)
    print(f"Stereo XMP metadata file created: {xmp_file_path}")

def add_xmp_to_heic(heic_path, xmp_file_path, output_heic_with_metadata):
    """
    Uses ExifTool to embed XMP metadata from the XMP file into a HEIC file.
    
    Args:
        heic_path (str): Path to the original HEIC file.
        xmp_file_path (str): Path to the XMP metadata file.
        output_heic_with_metadata (str): Path for the output HEIC file with embedded metadata.
    """
    command = [
        "exiftool",
        f"-xmp<={xmp_file_path}",
        "-o", output_heic_with_metadata,
        heic_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"XMP metadata embedded successfully: {output_heic_with_metadata}")
    else:
        print(f"Error embedding XMP metadata: {result.stderr}")

if __name__ == "__main__":
    # Update these paths as needed.
    heic_path = "C:/Users/adity/OneDrive/Desktop/samsung 2.0/5_stereo.heic"  # Your side-by-side (SBS) HEIC image.
    xmp_file_path = "stereo_output.xmp"
    output_heic_with_metadata = "sbs_with_metadata.heic"
    
    # Define the stereo mode. Here, "SBS" denotes a side-by-side stereo image.
    create_stereo_xmp("SBS", xmp_file_path)
    
    # Embed the stereo metadata into the HEIC file.
    add_xmp_to_heic(heic_path, xmp_file_path, output_heic_with_metadata)
